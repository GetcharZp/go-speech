package speech

import (
	"fmt"
	ort "github.com/yalue/onnxruntime_go"
	"runtime"
	"sync"
)

type OnnxConfig struct {
	SessionOptions *ort.SessionOptions

	// 必填参数
	OnnxRuntimeLibPath string // onnxruntime.dll (或 .so, .dylib) 的路径
	// 可选参数
	UseCuda    bool // (可选) 是否启用 CUDA
	NumThreads int  // (可选) ONNX 线程数, 默认由CPU核心数决定

	// EnableCpuMemArena 控制 ONNX 的内存池策略
	// false (默认): 禁用内存池，推理速度稍慢，但 Destroy 后立即归还内存给 OS ，解决内存滞留问题
	// true: 启用内存池，推理速度最快，但 Destroy 后内存会被缓存以供复用
	EnableCpuMemArena bool
}

var (
	initErr error
	once    sync.Once
)

// New 初始化 ONNX 环境
func (cfg *OnnxConfig) New() error {
	// 初始化 ONNX Runtime
	if cfg.OnnxRuntimeLibPath == "" {
		return fmt.Errorf("OnnxRuntimeLibPath 不能为空")
	}
	once.Do(func() {
		ort.SetSharedLibraryPath(cfg.OnnxRuntimeLibPath)
		initErr = ort.InitializeEnvironment()
	})
	if initErr != nil {
		return fmt.Errorf("初始化 ONNX Runtime 环境失败: %w", initErr)
	}

	// 创建会话选项 (设置线程)
	options, err := ort.NewSessionOptions()
	if err != nil {
		return err
	}
	if cfg.NumThreads > 0 {
		if err := options.SetIntraOpNumThreads(cfg.NumThreads); err != nil {
			return err
		}
	}

	// 设置内存策略
	if err := options.SetCpuMemArena(cfg.EnableCpuMemArena); err != nil {
		return fmt.Errorf("设置 CPU 内存池失败: %w", err)
	}

	// 启用CUDA
	if cfg.UseCuda {
		cudaOptions, err := ort.NewCUDAProviderOptions()
		if err != nil {
			return fmt.Errorf("创建 CUDAProviderOptions 失败: %w", err)
		}
		defer cudaOptions.Destroy()
		if err := options.AppendExecutionProviderCUDA(cudaOptions); err != nil {
			return fmt.Errorf("添加 CUDA 执行提供者失败: %w", err)
		}
	}
	cfg.SessionOptions = options

	return nil
}

// DefaultLibraryPath 根据运行时环境判断加载哪个库文件
func DefaultLibraryPath() string {
	baseDir := "./lib/"
	libName := "onnxruntime"

	// windows onnxruntime.dll
	if runtime.GOOS == "windows" {
		return baseDir + libName + ".dll"
	}

	// linux darwin ext
	var ext string
	switch runtime.GOOS {
	case "darwin":
		ext = "dylib"
	case "linux":
		ext = "so"
	default:
		return baseDir + libName + "_amd64.so" // 默认返回 linux amd64
	}

	// 拼接完整路径: ./lib/onnxruntime + _ + amd64/arm64 + . + so/dylib
	return fmt.Sprintf("%s%s_%s.%s", baseDir, libName, runtime.GOARCH, ext)
}
