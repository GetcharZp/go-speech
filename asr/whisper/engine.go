package whisper

import (
	"fmt"
	"github.com/getcharzp/go-speech"
	ort "github.com/getcharzp/onnxruntime_purego"
	"github.com/up-zero/gotool/convertutil"
	"math"
	"os"
	"strings"
	"unicode/utf8"
)

// Engine 封装了 Whisper 的 ONNX 运行时和相关资源
type Engine struct {
	encSession  *ort.Session
	decSession  *ort.Session
	tokenMap    map[int]string
	addTokenMap map[string]int

	modelLayers int
	maxTokens   int
	numHeads    int
	headDim     int

	sot, eot, noTime int

	decInputNames  []string
	decOutputNames []string
}

// NewEngine 初始化 Whisper 引擎
func NewEngine(cfg Config) (*Engine, error) {
	oc := new(speech.OnnxConfig)
	_ = convertutil.CopyProperties(cfg, oc)

	// 初始化 ONNX
	if err := oc.New(); err != nil {
		return nil, err
	}

	// 创建 Encoder 会话
	encSession, err := oc.OnnxEngine.NewSession(cfg.EncoderModelPath, oc.SessionOptions)
	if err != nil {
		return nil, fmt.Errorf("创建 Encoder 会话失败: %w", err)
	}

	// 组装 Decoder 的输入输出
	decInputNames := []string{"input_ids", "encoder_hidden_states", "use_cache_branch"}
	decOutputNames := []string{"logits"}
	for i := 0; i < cfg.ModelLayers; i++ {
		baseIn := fmt.Sprintf("past_key_values.%d", i)
		decInputNames = append(decInputNames,
			baseIn+".decoder.key", baseIn+".decoder.value",
			baseIn+".encoder.key", baseIn+".encoder.value",
		)
		baseOut := fmt.Sprintf("present.%d", i)
		decOutputNames = append(decOutputNames,
			baseOut+".decoder.key", baseOut+".decoder.value",
			baseOut+".encoder.key", baseOut+".encoder.value",
		)
	}

	// 创建 Decoder 会话
	decSession, err := oc.OnnxEngine.NewSession(cfg.DecoderModelPath, oc.SessionOptions)
	if err != nil {
		encSession.Destroy()
		return nil, fmt.Errorf("创建 Decoder 会话失败: %w", err)
	}

	// 加载 Token
	tokenMap, addTokenMap, err := loadTokens(cfg.TokensPath, cfg.AddedTokensPath)
	if err != nil {
		return nil, err
	}

	return &Engine{
		encSession:     encSession,
		decSession:     decSession,
		tokenMap:       tokenMap,
		addTokenMap:    addTokenMap,
		modelLayers:    cfg.ModelLayers,
		maxTokens:      cfg.MaxTokens,
		numHeads:       calculateNumHeads(cfg.ModelLayers),
		headDim:        64,
		decInputNames:  decInputNames,
		decOutputNames: decOutputNames,
		sot:            50258,
		eot:            50257,
		noTime:         50363,
	}, nil
}

// TranscribeFile 读取并转录 WAV 文件
//
// # Params:
//
//	wavPath: 音频文件路径
//	opt: 转录可选参数
func (e *Engine) TranscribeFile(wavPath string, opt ...TranscribeOption) (string, error) {
	wavBytes, err := os.ReadFile(wavPath)
	if err != nil {
		return "", fmt.Errorf("无法读取文件: %w", err)
	}
	return e.TranscribeBytes(wavBytes, opt...)
}

// TranscribeBytes 转录 WAV 字节流
//
// # Params:
//
//	wavBytes: 音频文件字节流
//	opt: 转录可选参数
func (e *Engine) TranscribeBytes(wavBytes []byte, opt ...TranscribeOption) (string, error) {
	samples, err := parseWavBytes(wavBytes)
	if err != nil {
		return "", fmt.Errorf("无法将 PCM 数据转换为 float32: %v", err)
	}
	return e.Transcribe(samples, opt...)
}

// Transcribe 对 float32 音频样本数据进行转录
//
// # Params:
//
//	samples: 采样率 16KHz 的单声道音频数据，范围 [-1, 1]
//	opt: 转录可选参数
func (e *Engine) Transcribe(samples []float32, opt ...TranscribeOption) (string, error) {
	// 特征提取
	features, err := e.extractFeatures(samples)
	if err != nil {
		return "", err
	}

	encIn, _ := ort.NewTensor([]int64{1, 80, 3000}, features)
	defer encIn.Destroy()

	inputValues := map[string]*ort.Value{
		"input_features": encIn,
	}

	// Encoder 推理
	outputValues, err := e.encSession.Run(inputValues)
	if err != nil {
		return "", fmt.Errorf("编码推理失败: %w", err)
	}
	outputValue := outputValues["last_hidden_state"]
	defer outputValue.Destroy()

	return e.runMergedDecoder(outputValue, opt...)
}

// runMergedDecoder Merge Decoder 推理
func (e *Engine) runMergedDecoder(encHiddenState *ort.Value, opt ...TranscribeOption) (string, error) {
	// prompt: [<|startoftranscript|>, <|language|>, <|task|>, <|notimestamps|>]
	prompt := []int64{int64(e.sot)}
	if len(opt) == 0 {
		prompt = append(prompt, int64(50260)) // zh
		prompt = append(prompt, int64(50359)) // transcribe
	} else {
		if lang, ok := e.addTokenMap[fmt.Sprintf("<|%s|>", opt[0].Language)]; ok {
			prompt = append(prompt, int64(lang))
		} else {
			return "", fmt.Errorf("未知语言: %s", opt[0].Language)
		}

		if task, ok := e.addTokenMap[fmt.Sprintf("<|%s|>", opt[0].Task)]; ok {
			prompt = append(prompt, int64(task))
		} else {
			return "", fmt.Errorf("未知任务: %s", opt[0].Task)
		}

		prompt = append(prompt, int64(50260)) // zh
		prompt = append(prompt, int64(50359)) // transcribe
	}
	prompt = append(prompt, int64(e.noTime))

	pastTensors, err := e.createPastTensors()
	if err != nil {
		return "", err
	}

	inputIdsTensor, _ := ort.NewTensor([]int64{1, int64(len(prompt))}, prompt)
	useCacheTensor, _ := ort.NewTensor([]int64{1}, []bool{false})

	prefillInputs := map[string]*ort.Value{
		"input_ids":             inputIdsTensor,
		"encoder_hidden_states": encHiddenState,
		"use_cache_branch":      useCacheTensor,
	}
	for name, value := range pastTensors {
		prefillInputs[name] = value
	}

	outputs, err := e.decSession.Run(prefillInputs)
	if err != nil {
		inputIdsTensor.Destroy()
		useCacheTensor.Destroy()
		for _, t := range pastTensors {
			t.Destroy()
		}
		return "", fmt.Errorf("预解码推理失败: %w", err)
	}
	inputIdsTensor.Destroy()
	useCacheTensor.Destroy()
	for _, t := range pastTensors {
		t.Destroy()
	}

	logits := outputs["logits"]

	nextTokenID := e.sampleTokenPrefill(logits)
	logits.Destroy()

	// 更新缓存
	for name, t := range outputs {
		if strings.Contains(name, "present") {
			key := strings.ReplaceAll(name, "present", "past_key_values")
			pastTensors[key] = t
		}
	}

	generatedTokens := make([]int, 0)
	generatedTokens = append(generatedTokens, nextTokenID)

	loopInputs := make(map[string]*ort.Value, 3+len(pastTensors))

	// 循环生成
	for i := 0; i < e.maxTokens; i++ {
		if nextTokenID == e.eot {
			break
		}

		currInTensor, _ := ort.NewTensor([]int64{1, 1}, []int64{int64(nextTokenID)})
		currCacheTensor, _ := ort.NewTensor([]int64{1}, []bool{true})

		loopInputs["input_ids"] = currInTensor
		loopInputs["encoder_hidden_states"] = encHiddenState
		loopInputs["use_cache_branch"] = currCacheTensor
		for name, value := range pastTensors {
			loopInputs[name] = value
		}

		newOutputs, err := e.decSession.Run(loopInputs)
		currInTensor.Destroy()
		currCacheTensor.Destroy()
		if err != nil {
			return "", fmt.Errorf("第 %d 步解码推理失败: %w", i, err)
		}

		newLogits := newOutputs["logits"]

		// 更新 KV Cache
		for name, newValue := range newOutputs {
			if name == "logits" {
				continue
			}

			if strings.Contains(name, "decoder") {
				// 更新 Decoder 的 Key/Value (present.X.decoder.key/value)
				key := strings.ReplaceAll(name, "present", "past_key_values")

				// 销毁旧的缓存，存入新缓存
				if oldV, ok := pastTensors[key]; ok {
					oldV.Destroy()
				}
				pastTensors[key] = newValue
			} else {
				// encoder 部分直接销毁掉防止泄露
				newValue.Destroy()
			}
		}

		nextTokenID = e.sampleToken(newLogits, generatedTokens)
		newLogits.Destroy()

		generatedTokens = append(generatedTokens, nextTokenID)
	}

	// 清除缓存张量
	for _, t := range pastTensors {
		t.Destroy()
	}

	return e.decode(generatedTokens), nil
}

// createPastTensors 创建缓存张量
func (e *Engine) createPastTensors() (map[string]*ort.Value, error) {
	tensors := make(map[string]*ort.Value)
	shape := []int64{1, int64(e.numHeads), 0, 64}
	var empty = make([]float32, e.numHeads*64)

	for i := 3; i < len(e.decInputNames); i++ {
		name := e.decInputNames[i]
		t, err := ort.NewTensor(shape, empty)
		if err != nil {
			for _, v := range tensors {
				v.Destroy()
			}
			return nil, err
		}
		tensors[name] = t
	}

	return tensors, nil
}

// sampleTokenPrefill 获取词表中的最优 Token 索引（预解码）
func (e *Engine) sampleTokenPrefill(logits *ort.Value) int {
	data, _ := ort.GetTensorData[float32](logits)
	shape, _ := logits.GetShape()

	vocabSize := 51865

	if len(shape) < 2 {
		return e.eot
	}

	promptLen := int(shape[1])
	startIdx := (promptLen - 1) * vocabSize

	if startIdx < 0 || startIdx+vocabSize > len(data) {
		return e.eot
	}

	maxIdx := 0
	maxVal := float32(-math.MaxFloat32)

	for i := 0; i < vocabSize; i++ {
		score := data[startIdx+i]

		if i == e.noTime || i == e.sot || i == e.eot {
			score = -float32(math.MaxFloat32)
		}

		if i >= 50364 {
			score = -float32(math.MaxFloat32)
		}

		if score > maxVal {
			maxVal = score
			maxIdx = i
		}
	}

	return maxIdx
}

// sampleToken 获取词表中的最优 Token 索引（循环）
func (e *Engine) sampleToken(logits *ort.Value, history []int) int {
	data, _ := ort.GetTensorData[float32](logits)
	vocabSize := 51865
	startIdx := len(data) - vocabSize

	if startIdx < 0 || startIdx+vocabSize > len(data) {
		return e.eot
	}

	scores := make([]float32, vocabSize)
	copy(scores, data[startIdx:startIdx+vocabSize])

	// 标记特殊 Token
	if e.noTime >= 0 && e.noTime < vocabSize {
		scores[e.noTime] = -float32(math.MaxFloat32)
	}
	if e.sot >= 0 && e.sot < vocabSize {
		scores[e.sot] = -float32(math.MaxFloat32)
	}
	if len(history) < 10 && e.eot >= 0 && e.eot < vocabSize {
		scores[e.eot] = -float32(math.MaxFloat32)
	}

	// 重复惩罚
	if len(history) > 0 {
		lastToken := history[len(history)-1]
		if lastToken >= 0 && lastToken < vocabSize && scores[lastToken] > -float32(math.MaxFloat32)/2 {
			scores[lastToken] -= 1.0
		}
		windowStart := 0
		if len(history) > 5 {
			windowStart = len(history) - 5
		}
		for _, histId := range history[windowStart:] {
			if histId >= 0 && histId < vocabSize && scores[histId] > -float32(math.MaxFloat32)/2 {
				scores[histId] -= 0.5
			}
		}
	}

	maxIdx := 0
	maxVal := float32(-math.MaxFloat32)

	for i := 0; i < vocabSize; i++ {
		score := scores[i]
		if score > maxVal {
			maxVal = score
			maxIdx = i
		}
	}

	return maxIdx
}

// decode Token ids 转为文本
func (e *Engine) decode(ids []int) string {
	initByteDecoder()
	var buf []byte

	for _, id := range ids {
		if id == e.eot {
			break
		}

		// 跳过时长
		if id >= 50364 {
			continue
		}
		// 跳过其他特殊 Token
		if id >= 50257 && id < 50364 {
			continue
		}

		s, ok := e.tokenMap[id]
		if !ok {
			continue
		}

		// 字节解码
		for _, r := range s {
			if b, ok := byteDecoder[r]; ok {
				buf = append(buf, b)
			} else {
				buf = append(buf, byte(r))
			}
		}
	}

	// 验证 UTF-8
	text := string(buf)
	if !utf8.ValidString(text) {
		text = strings.ToValidUTF8(text, "")
	}

	return strings.TrimSpace(text)
}

// Destroy 释放相关资源
func (e *Engine) Destroy() error {
	if e.encSession != nil {
		e.encSession.Destroy()
	}
	if e.decSession != nil {
		e.decSession.Destroy()
	}
	return nil
}
