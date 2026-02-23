package pipertts

import (
	"fmt"
	"github.com/getcharzp/go-speech"
	ort "github.com/getcharzp/onnxruntime_purego"
	"github.com/up-zero/gotool/convertutil"
	"github.com/up-zero/gotool/mediautil"
)

// Engine Piper-TTS 引擎结构
type Engine struct {
	session     *ort.Session
	piperConfig PiperConfig
	config      Config
}

// NewEngine 初始化 Piper 引擎
func NewEngine(cfg Config) (*Engine, error) {
	oc := new(speech.OnnxConfig)
	_ = convertutil.CopyProperties(cfg, oc)

	if err := oc.New(); err != nil {
		return nil, err
	}

	// 读取 Piper 专属配置 (.onnx.json)
	piperCfg, err := loadPiperConfig(cfg.ConfigPath)
	if err != nil {
		return nil, fmt.Errorf("加载 Piper 配置失败: %w", err)
	}

	// 创建 ONNX 会话
	session, err := oc.OnnxEngine.NewSession(cfg.ModelPath, oc.SessionOptions)
	if err != nil {
		return nil, fmt.Errorf("创建会话失败: %w", err)
	}

	return &Engine{
		session:     session,
		piperConfig: piperCfg,
		config:      cfg,
	}, nil
}

// Synthesize 合成 PCM 数据
func (e *Engine) Synthesize(text string) ([]float32, error) {
	// 文本标准化
	text = convertutil.TextToChinese(text)

	inputIDs := e.textToIds(text)
	if len(inputIDs) == 0 {
		return nil, fmt.Errorf("音素序列转换结果为空")
	}

	return e.runInference(inputIDs)
}

// SynthesizeToWav 合成并导出为 WAV 字节流
func (e *Engine) SynthesizeToWav(phonemesText string) ([]byte, error) {
	pcmData, err := e.Synthesize(phonemesText)
	if err != nil {
		return nil, err
	}

	return mediautil.Float32ToWavBytes(pcmData, e.piperConfig.Audio.SampleRate, channels, bitsPerSample)
}

// runInference 执行 ONNX 推理
func (e *Engine) runInference(inputIDs []int64) ([]float32, error) {
	seqLength := int64(len(inputIDs))

	// input [1, phonemes]
	tInput, err := ort.NewTensor([]int64{1, seqLength}, inputIDs)
	if err != nil {
		return nil, fmt.Errorf("构建 input 失败: %w", err)
	}
	defer tInput.Destroy()

	// input_lengths [1] [seqLength]
	tInputLengths, err := ort.NewTensor([]int64{1}, []int64{seqLength})
	if err != nil {
		return nil, fmt.Errorf("构建 input_lengths 失败: %w", err)
	}
	defer tInputLengths.Destroy()

	// scales [3] [noise_scale, length_scale, noise_w]
	scalesData := []float32{
		e.piperConfig.Inference.NoiseScale,
		e.piperConfig.Inference.LengthScale,
		e.piperConfig.Inference.NoiseW,
	}
	tScales, err := ort.NewTensor([]int64{3}, scalesData)
	if err != nil {
		return nil, fmt.Errorf("构建 scales 失败: %w", err)
	}
	defer tScales.Destroy()

	inputValues := map[string]*ort.Value{
		"input":         tInput,
		"input_lengths": tInputLengths,
		"scales":        tScales,
	}

	outputValues, err := e.session.Run(inputValues)
	if err != nil {
		return nil, fmt.Errorf("piper 推理失败: %w", err)
	}

	defer func() {
		for _, v := range outputValues {
			if v != nil {
				v.Destroy()
			}
		}
	}()

	outputValue := outputValues["output"]

	rawData, err := ort.GetTensorData[float32](outputValue)
	if err != nil {
		return nil, fmt.Errorf("读取输出数据失败: %w", err)
	}

	result := make([]float32, len(rawData))
	copy(result, rawData)
	return result, nil
}

// Destroy 释放资源
func (e *Engine) Destroy() {
	if e.session != nil {
		e.session.Destroy()
	}
}
