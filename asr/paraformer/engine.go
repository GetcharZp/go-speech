package paraformer

import (
	"fmt"
	"github.com/getcharzp/go-speech"
	ort "github.com/getcharzp/onnxruntime_purego"
	"github.com/up-zero/gotool/convertutil"
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

	return &Engine{
		session:  session,
		tokenMap: tokenMap,
		negMean:  negMean,
		invStd:   invStd,
	}, nil
}

// Destroy 释放相关资源
func (e *Engine) Destroy() {
	if e.session != nil {
		e.session.Destroy()
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
	text := e.decode(tokenIDs)
	return text, nil
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
func (e *Engine) decode(ids []int) string {
	var sb strings.Builder
	for _, idx := range ids {
		if word, ok := e.tokenMap[idx]; ok {
			if word == "<blank>" || word == "<s>" || word == "</s>" || word == "<unk>" {
				sb.WriteByte(' ')
			} else if strings.HasSuffix(word, "@@") {
				word = strings.ReplaceAll(word, "@@", "")
				sb.WriteString(word)
			} else {
				sb.WriteString(word)
				sb.WriteByte(' ')
			}
		}
	}
	return sb.String()
}
