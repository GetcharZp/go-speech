package paraformer

import (
	"fmt"
	"github.com/getcharzp/go-speech"
	"github.com/up-zero/gotool/convertutil"
	ort "github.com/yalue/onnxruntime_go"
	"math"
	"os"
	"strings"
)

// Engine 封装了 Paraformer ASR 的 ONNX 运行时和相关资源
type Engine struct {
	session  *ort.DynamicAdvancedSession
	tokenMap map[int]string
	negMean  []float32 // CMVN 均值
	invStd   []float32 // CMVN 方差倒数
}

// NewEngine 初始化 Paraformer ASR 引擎
func NewEngine(cfg Config) (*Engine, error) {
	onnxConfig := new(speech.OnnxConfig)
	if err := convertutil.CopyProperties(cfg, onnxConfig); err != nil {
		return nil, fmt.Errorf("复制参数失败: %w", err)
	}
	// 初始化 ONNX
	if err := onnxConfig.New(); err != nil {
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
	inputNames := []string{"speech", "speech_lengths"}
	outputNames := []string{"logits"}
	session, err := ort.NewDynamicAdvancedSession(
		cfg.ModelPath,
		inputNames,
		outputNames,
		onnxConfig.SessionOptions,
	)
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
func (p *Engine) Destroy() error {
	if p.session != nil {
		return p.session.Destroy()
	}
	return nil
}

// TranscribeFile 读取 WAV 文件并进行语音识别
//
// # Params:
//
//	wavPath: 音频文件路径
func (p *Engine) TranscribeFile(wavPath string) (string, error) {
	wavBytes, err := os.ReadFile(wavPath)
	if err != nil {
		return "", fmt.Errorf("无法读取文件: %v", err)
	}
	return p.TranscribeBytes(wavBytes)
}

// TranscribeBytes 读取 WAV 字节流并进行语音识别
//
// # Params:
//
//	wavBytes: 音频文件字节流
func (p *Engine) TranscribeBytes(wavBytes []byte) (string, error) {
	samples, err := parseWavBytes(wavBytes)
	if err != nil {
		return "", fmt.Errorf("无法将 PCM 数据转换为 float32: %v", err)
	}
	return p.Transcribe(samples)
}

// Transcribe 对 float32 音频样本数据进行识别
//
// # Params:
//
//	samples: 采样率为 16KHz 的单声道音频数据，范围 [-1, 1]
func (p *Engine) Transcribe(samples []float32) (string, error) {
	if len(samples) == 0 {
		return "", fmt.Errorf("输入的音频数据为空")
	}

	// 特征提取
	features, featLen, err := p.extractFeatures(samples)
	if err != nil {
		return "", err
	}

	// 推理
	tokenIDs, err := p.runInference(features, featLen)
	if err != nil {
		return "", err
	}

	// 解码
	text := p.decode(tokenIDs)
	return text, nil
}

// runInference 推理
func (p *Engine) runInference(features []float32, featLen int32) ([]int, error) {
	// 构建张量
	tSpeech, err := ort.NewTensor(ort.NewShape(1, int64(featLen), 560), features)
	if err != nil {
		return nil, fmt.Errorf("创建 speech tensor 失败: %w", err)
	}
	defer tSpeech.Destroy()
	tLen, err := ort.NewTensor(ort.NewShape(1), []int32{featLen})
	if err != nil {
		return nil, fmt.Errorf("创建 length tensor 失败: %w", err)
	}
	defer tLen.Destroy()

	inputs := []ort.Value{tSpeech, tLen}
	outputs := make([]ort.Value, 1)

	// 执行
	if err := p.session.Run(inputs, outputs); err != nil {
		return nil, fmt.Errorf("推理运行失败: %w", err)
	}
	defer outputs[0].Destroy()

	// 获取结果
	resultTensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("推理输出类型断言失败，期望 *Tensor[float32]")
	}

	rawData := resultTensor.GetData()
	outputShape := resultTensor.GetShape() // [1, T_out, TokenSize]
	if len(outputShape) < 3 {
		return nil, fmt.Errorf("输出结果维度异常: %+v", outputShape)
	}

	return getTokenIds(rawData, int(outputShape[1]), int(outputShape[2])), nil
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
func (p *Engine) decode(ids []int) string {
	var sb strings.Builder
	for _, idx := range ids {
		if word, ok := p.tokenMap[idx]; ok {
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
