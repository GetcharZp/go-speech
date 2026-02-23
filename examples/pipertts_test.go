package examples

import (
	"github.com/up-zero/gotool/fileutil"
	"testing"

	"github.com/getcharzp/go-speech/tts/pipertts"
)

func TestPiperTTS(t *testing.T) {
	cfg := pipertts.Config{
		OnnxRuntimeLibPath: "../lib/onnxruntime.dll",
		ModelPath:          "../pipertts_weights/zh_CN-xiao_ya-medium.onnx",
		ConfigPath:         "../pipertts_weights/zh_CN-xiao_ya-medium.onnx.json",
	}

	ttsEngine, err := pipertts.NewEngine(cfg)
	if err != nil {
		t.Fatalf("创建引擎失败: %v", err)
	}
	defer ttsEngine.Destroy()

	testText := "2019年12月30日，中国人口突破14亿人。联系电话: 13800138000。"
	wavBytes, err := ttsEngine.SynthesizeToWav(testText)
	if err != nil {
		t.Fatalf("合成失败: %v", err)
	}

	outputPath := "pipertts_output.wav"
	err = fileutil.FileSave(outputPath, wavBytes)
	if err != nil {
		t.Fatalf("保存失败: %v", err)
	}
}
