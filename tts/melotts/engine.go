package melotts

import (
	"fmt"
	"github.com/getcharzp/go-speech"
	ort "github.com/getcharzp/onnxruntime_purego"
	"github.com/up-zero/gotool/convertutil"
	"github.com/up-zero/gotool/mediautil"
)

// Engine 封装了 MeloTTS 的 ONNX 运行时和相关资源
// 不仅持有模型会话，还缓存了分词器和词典数据
type Engine struct {
	session  *ort.Session
	lexicon  map[string]LexiconItem
	tokenMap map[string]int64
	config   Config
}

// NewEngine 初始化 MeloTTS 引擎
func NewEngine(cfg Config) (*Engine, error) {
	oc := new(speech.OnnxConfig)
	_ = convertutil.CopyProperties(cfg, oc)

	// 初始化 ONNX
	if err := oc.New(); err != nil {
		return nil, err
	}

	if cfg.ModelPath == "" || cfg.TokenPath == "" || cfg.LexiconPath == "" {
		return nil, fmt.Errorf("模型、Tokens 和 Lexicon 文件路径不能为空")
	}

	// 加载资源 (Tokens 和 Lexicon)
	tokenMap, err := loadTokens(cfg.TokenPath)
	if err != nil {
		return nil, fmt.Errorf("加载 Tokens 失败: %w", err)
	}
	lexicon, err := loadLexicon(cfg.LexiconPath)
	if err != nil {
		return nil, fmt.Errorf("加载 Lexicon 失败: %w", err)
	}

	// 创建 ONNX 会话
	session, err := oc.OnnxEngine.NewSession(cfg.ModelPath, oc.SessionOptions)
	if err != nil {
		return nil, fmt.Errorf("创建 ONNX 会话失败: %w", err)
	}

	return &Engine{
		session:  session,
		lexicon:  lexicon,
		tokenMap: tokenMap,
		config:   cfg,
	}, nil
}

// Synthesize 将文本转换为语音数据 (float32 PCM)
//
// # Params:
//
//	text: 需要转换的文本
//	speed: 语速调节,数值越大越快,1.0为正常语速
func (e *Engine) Synthesize(text string, speed float32) ([]float32, error) {
	// 文本标准化
	normalizedText := convertutil.TextToChinese(text)

	// 文本转 ID (G2P)
	inputIDs, toneIDs, err := e.textToIds(normalizedText)
	if err != nil {
		return nil, fmt.Errorf("G2P 转换失败: %w", err)
	}

	// 执行 ONNX 推理
	return e.runInference(inputIDs, toneIDs, speed)
}

// SynthesizeToWav 将文本转换为 WAV 格式的字节流
//
// # Params:
//
//	text: 需要转换的文本
//	speed: 语速调节,数值越大越快,1.0为正常语速
func (e *Engine) SynthesizeToWav(text string, speed float32) ([]byte, error) {
	pcmData, err := e.Synthesize(text, speed)
	if err != nil {
		return nil, err
	}

	return mediautil.Float32ToWavBytes(pcmData, SampleRate, channels, bitsPerSample)
}

// Destroy 释放相关资源
func (e *Engine) Destroy() {
	if e.session != nil {
		e.session.Destroy()
	}
}

// runInference 推理
func (e *Engine) runInference(inputIDs []int64, toneIDs []int64, speed float32) ([]float32, error) {
	seqLength := int64(len(inputIDs))

	// 构建张量
	tX, err := ort.NewTensor([]int64{1, seqLength}, inputIDs)
	if err != nil {
		return nil, fmt.Errorf("创建 X tensor 失败: %w", err)
	}
	defer tX.Destroy()
	tLen, err := ort.NewTensor([]int64{1}, []int64{seqLength})
	if err != nil {
		return nil, fmt.Errorf("创建 length tensor 失败: %w", err)
	}
	defer tLen.Destroy()
	tTones, err := ort.NewTensor([]int64{1, seqLength}, toneIDs)
	if err != nil {
		return nil, fmt.Errorf("创建 tones tensor 失败: %w", err)
	}
	defer tTones.Destroy()
	tSid, err := ort.NewTensor([]int64{1}, []int64{speakerID})
	if err != nil {
		return nil, fmt.Errorf("创建 sid tensor 失败: %w", err)
	}
	defer tSid.Destroy()

	// 参数控制
	// noise_scale (0.667), length_scale (1.0 / speed), noise_scale_w (0.8)
	// length_scale 控制语速，值越大语速越慢，所以用 1.0/speed
	noiseScale := float32(0.667)
	lengthScale := float32(1.0)
	if speed > 0 {
		lengthScale = 1.0 / speed
	}
	noiseScaleW := float32(0.8)

	tNoise, err := ort.NewTensor([]int64{1}, []float32{noiseScale})
	if err != nil {
		return nil, fmt.Errorf("创建 noise_scale tensor 失败: %w", err)
	}
	defer tNoise.Destroy()
	tLScale, err := ort.NewTensor([]int64{1}, []float32{lengthScale})
	if err != nil {
		return nil, fmt.Errorf("创建 length_scale tensor 失败: %w", err)
	}
	defer tLScale.Destroy()
	tNoiseW, err := ort.NewTensor([]int64{1}, []float32{noiseScaleW})
	if err != nil {
		return nil, fmt.Errorf("创建 noise_scale_w tensor 失败: %w", err)
	}
	defer tNoiseW.Destroy()

	inputValues := map[string]*ort.Value{
		"x":             tX,
		"x_lengths":     tLen,
		"tones":         tTones,
		"sid":           tSid,
		"noise_scale":   tNoise,
		"length_scale":  tLScale,
		"noise_scale_w": tNoiseW,
	}

	// 执行
	outputValues, err := e.session.Run(inputValues)
	if err != nil {
		return nil, fmt.Errorf("推理运行失败: %w", err)
	}
	outputValue := outputValues["y"]
	defer outputValue.Destroy()

	// 获取结果
	rawData, err := ort.GetTensorData[float32](outputValue)
	if err != nil {
		return nil, fmt.Errorf("获取输出数据失败: %w", err)
	}

	// 拷贝数据
	result := make([]float32, len(rawData))
	copy(result, rawData)

	return result, nil
}
