package paraformer

import (
	"encoding/json"
	"fmt"
	"github.com/getcharzp/go-speech"
	ort "github.com/getcharzp/onnxruntime_purego"
	"github.com/up-zero/gotool/convertutil"
	"github.com/up-zero/gotool/validator"
	"math"
	"os"
	"strings"
)

// Engine 封装了 Paraformer ASR 的 ONNX 运行时和相关资源
type Engine struct {
	session  *ort.Session
	tokenMap map[int]string
	negMean  []float32 // CMVN 均值
	invStd   []float32 // CMVN 方差倒数

	// 标点模型相关
	punctuationSession  *ort.Session
	punctuationTokenMap map[string]int // 文本 -> ID
	punctuationList     []string       // 标点符号
}

// NewEngine 初始化 Paraformer ASR 引擎
func NewEngine(cfg Config) (*Engine, error) {
	oc := new(speech.OnnxConfig)
	_ = convertutil.CopyProperties(cfg, oc)

	// 初始化 ONNX
	if err := oc.New(); err != nil {
		return nil, err
	}

	// 加载资源 (Tokens 和 CMVN)
	tokenMap, err := loadTokens(cfg.TokensPath)
	if err != nil {
		return nil, fmt.Errorf("加载词表失败: %w", err)
	}
	negMean, invStd, err := loadCMVN(cfg.CMVNPath)
	if err != nil {
		// 某些模型可能不强制需要 CMVN，这里根据需求决定是报错还是警告
		return nil, fmt.Errorf("加载 CMVN 失败: %w", err)
	}

	// 创建 ONNX 会话
	session, err := oc.OnnxEngine.NewSession(cfg.ModelPath, oc.SessionOptions)
	if err != nil {
		return nil, fmt.Errorf("创建 ONNX 会话失败: %w", err)
	}

	engine := &Engine{
		session:  session,
		tokenMap: tokenMap,
		negMean:  negMean,
		invStd:   invStd,
	}

	// 加载标点模型
	if cfg.PunctuationModelPath != "" && cfg.PunctuationTokensPath != "" {
		// 加载标点词表 tokens.json
		pTokenMap, err := loadPunctuationTokens(cfg.PunctuationTokensPath)
		if err != nil {
			return nil, fmt.Errorf("加载标点词表失败: %w", err)
		}
		engine.punctuationTokenMap = pTokenMap
		// CT-Transformer 标点定义
		// 0:<unk>, 1:_, 2:，, 3:。, 4:？, 5:、
		engine.punctuationList = []string{"", "", "，", "。", "？", "、"}

		// 创建标点模型会话
		pSession, err := oc.OnnxEngine.NewSession(cfg.PunctuationModelPath, oc.SessionOptions)
		if err != nil {
			return nil, err
		}
		engine.punctuationSession = pSession
	}

	return engine, nil
}

// Destroy 释放相关资源
func (e *Engine) Destroy() {
	if e.session != nil {
		e.session.Destroy()
	}
	if e.punctuationSession != nil {
		e.punctuationSession.Destroy()
	}
}

// TranscribeFile 读取 WAV 文件并进行语音识别
//
// # Params:
//
//	wavPath: 音频文件路径
func (e *Engine) TranscribeFile(wavPath string) (string, error) {
	wavBytes, err := os.ReadFile(wavPath)
	if err != nil {
		return "", fmt.Errorf("无法读取文件: %v", err)
	}
	return e.TranscribeBytes(wavBytes)
}

// TranscribeBytes 读取 WAV 字节流并进行语音识别
//
// # Params:
//
//	wavBytes: 音频文件字节流
func (e *Engine) TranscribeBytes(wavBytes []byte) (string, error) {
	samples, err := parseWavBytes(wavBytes)
	if err != nil {
		return "", fmt.Errorf("无法将 PCM 数据转换为 float32: %v", err)
	}
	return e.Transcribe(samples)
}

// Transcribe 对 float32 音频样本数据进行识别
//
// # Params:
//
//	samples: 采样率为 16KHz 的单声道音频数据，范围 [-1, 1]
func (e *Engine) Transcribe(samples []float32) (string, error) {
	if len(samples) == 0 {
		return "", fmt.Errorf("输入的音频数据为空")
	}

	// 特征提取
	features, featLen, err := e.extractFeatures(samples)
	if err != nil {
		return "", err
	}

	// 推理
	tokenIDs, err := e.runInference(features, featLen)
	if err != nil {
		return "", err
	}

	// 解码
	words := e.decode(tokenIDs)

	// 标点预测
	if e.punctuationSession != nil {
		words, err = e.runPunctuationInference(words)
		if err != nil {
			return "", err
		}
	}

	return e.join(words), nil
}

// runInference 推理
func (e *Engine) runInference(features []float32, featLen int32) ([]int, error) {
	// 构建张量
	tSpeech, err := ort.NewTensor([]int64{1, int64(featLen), 560}, features)
	if err != nil {
		return nil, fmt.Errorf("创建 speech tensor 失败: %w", err)
	}
	defer tSpeech.Destroy()
	tLen, err := ort.NewTensor([]int64{1}, []int32{featLen})
	if err != nil {
		return nil, fmt.Errorf("创建 length tensor 失败: %w", err)
	}
	defer tLen.Destroy()

	inputValues := map[string]*ort.Value{
		"speech":         tSpeech,
		"speech_lengths": tLen,
	}

	// 执行
	outputValues, err := e.session.Run(inputValues)
	if err != nil {
		return nil, fmt.Errorf("推理运行失败: %w", err)
	}
	outputValue := outputValues["logits"]
	defer outputValue.Destroy()

	// 获取结果
	data, err := ort.GetTensorData[float32](outputValue)
	if err != nil {
		return nil, fmt.Errorf("获取输出数据失败: %w", err)
	}

	outputShape, err := outputValue.GetShape() // [1, T_out, TokenSize]
	if err != nil {
		return nil, fmt.Errorf("输出结果维度异常: %w", err)
	}

	return getTokenIds(data, int(outputShape[1]), int(outputShape[2])), nil
}

// 获取 token ids
func getTokenIds(tokenScores []float32, steps int, tokenSize int) []int {
	ids := make([]int, 0, steps)
	for t := 0; t < steps; t++ {
		start := t * tokenSize
		end := start + tokenSize
		if end > len(tokenScores) {
			break
		}

		// 获取当前概率最大的 token 索引
		var maxIdx int
		var maxVal float32 = -math.MaxFloat32
		curStepScores := tokenScores[start:end]
		for i, val := range curStepScores {
			if val > maxVal {
				maxVal = val
				maxIdx = i
			}
		}
		ids = append(ids, maxIdx)
	}
	return ids
}

// decode 解码，将 token ids 转换为文本
func (e *Engine) decode(ids []int) []string {
	var words []string
	var currentWord strings.Builder

	for _, idx := range ids {
		if word, ok := e.tokenMap[idx]; ok {
			if word == "<blank>" || word == "<s>" || word == "</s>" || word == "<unk>" {
				continue
			} else if strings.HasSuffix(word, "@@") {
				currentWord.WriteString(strings.ReplaceAll(word, "@@", ""))
			} else {
				currentWord.WriteString(word)
				words = append(words, currentWord.String())
				currentWord.Reset()
			}
		}
	}
	if currentWord.Len() > 0 {
		words = append(words, currentWord.String())
	}
	return words
}

// loadPunctuationTokens 加载标点 JSON 词表
func loadPunctuationTokens(path string) (map[string]int, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var tokenList []string
	if err := json.Unmarshal(data, &tokenList); err != nil {
		return nil, err
	}

	tokenMap := make(map[string]int, len(tokenList))
	for i, token := range tokenList {
		tokenMap[token] = i
	}
	return tokenMap, nil
}

// runPunctuationInference 执行标点预测
func (e *Engine) runPunctuationInference(words []string) ([]string, error) {
	if e.punctuationSession == nil || len(words) == 0 {
		return []string{}, nil
	}

	// 获取标点模型的输入ID
	inputIds := make([]int32, len(words))
	for i, w := range words {
		if id, ok := e.punctuationTokenMap[w]; ok {
			inputIds[i] = int32(id)
		} else {
			inputIds[i] = int32(e.punctuationTokenMap["<unk>"])
		}
	}

	tInputs, _ := ort.NewTensor([]int64{1, int64(len(inputIds))}, inputIds)
	defer tInputs.Destroy()

	tLengths, _ := ort.NewTensor([]int64{1}, []int32{int32(len(inputIds))})
	defer tLengths.Destroy()

	// 推理
	outputValues, err := e.punctuationSession.Run(map[string]*ort.Value{
		"inputs":       tInputs,
		"text_lengths": tLengths,
	})
	if err != nil {
		return []string{}, err
	}
	tLogits := outputValues["logits"]
	defer tLogits.Destroy()

	// 解析结果 [1, N, 6]
	data, _ := ort.GetTensorData[float32](tLogits)
	shape, _ := tLogits.GetShape()
	numSteps := int(shape[1])
	numClasses := int(shape[2])

	var newWords []string
	for i := 0; i < numSteps; i++ {
		offset := i * numClasses
		newWords = append(newWords, words[i])

		// 计算当前位置概率最大的标点索引
		maxIdx := 0
		var maxVal float32 = -1e9
		for j := 0; j < numClasses; j++ {
			val := data[offset+j]
			if val > maxVal {
				maxVal = val
				maxIdx = j
			}
		}

		if maxIdx > 0 && maxIdx < len(e.punctuationList) && maxIdx != 1 {
			newWords = append(newWords, e.punctuationList[maxIdx])
		}
	}
	return newWords, nil
}

// join 单词拼接
func (e *Engine) join(words []string) string {
	var sb strings.Builder
	for i, w := range words {
		sb.WriteString(w)
		if i < len(words)-1 {
			if !validator.IsChinese(w) || !validator.IsChinese(words[i+1]) {
				sb.WriteByte(' ')
			}
		}
	}
	return sb.String()
}
