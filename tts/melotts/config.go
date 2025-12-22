package melotts

import "github.com/getcharzp/go-speech"

const (
	// SampleRate 采样率，默认为 44100
	SampleRate = 44100
	// speakerID 说话人 ID
	speakerID = 1
	// channels 声道数
	channels = 1
	// bitsPerSample 采样位数
	bitsPerSample = 16
)

// Config 定义 MeloTTS 引擎的配置参数
type Config struct {
	// 必填参数
	OnnxRuntimeLibPath string // onnxruntime.dll (或 .so, .dylib) 的路径
	ModelPath          string // ONNX 模型路径
	TokenPath          string // tokens.txt 路径
	LexiconPath        string // lexicon.txt 路径

	// 可选参数
	UseCuda           bool // (可选) 是否启用 CUDA
	NumThreads        int  // (可选) ONNX 线程数, 默认由CPU核心数决定
	EnableCpuMemArena bool // (可选) 是否启用内存池
}

// DefaultConfig 返回一套默认的配置 (基于常见的目录结构)
func DefaultConfig() Config {
	return Config{
		OnnxRuntimeLibPath: speech.DefaultLibraryPath(),
		ModelPath:          "./melo_weights/model.onnx",
		TokenPath:          "./melo_weights/tokens.txt",
		LexiconPath:        "./melo_weights/lexicon.txt",
	}
}

// LexiconItem 存储音素和对应的声调信息
type LexiconItem struct {
	// Phones 音素
	Phones []string
	// Tones 音调
	Tones []int64
}
