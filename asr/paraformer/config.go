package paraformer

import "github.com/getcharzp/go-speech"

const (
	// sampleRate 采样率
	sampleRate = 16000
	// channels 声道数
	channels = 1
	// bitsPerSample 采样位数
	bitsPerSample = 16
)

// Config 定义 Paraformer 模型的配置参数
type Config struct {
	// 必填参数
	OnnxRuntimeLibPath string // onnxruntime.dll (或 .so, .dylib) 的路径
	ModelPath          string // ONNX 模型路径
	TokensPath         string // tokens.txt 路径
	CMVNPath           string // am.mvn 文件路径

	// 可选参数
	PunctuationModelPath  string // 标点模型路径
	PunctuationTokensPath string // 标点 tokens.json 路径
	UseCuda               bool   // (可选) 是否启用 CUDA
	NumThreads            int    // (可选) ONNX 线程数, 默认由CPU核心数决定
	EnableCpuMemArena     bool   // (可选) 是否启用内存池
}

// DefaultConfig 返回一套默认的配置 (基于常见的目录结构)
func DefaultConfig() Config {
	return Config{
		OnnxRuntimeLibPath: speech.DefaultLibraryPath(),
		ModelPath:          "./paraformer_weights/model.int8.onnx",
		TokensPath:         "./paraformer_weights/tokens.txt",
		CMVNPath:           "./paraformer_weights/am.mvn",
	}
}
