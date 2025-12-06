package examples

import (
	"github.com/getcharzp/go-speech/tts/melotts"
	"github.com/up-zero/gotool/fileutil"
	"testing"
)

func TestMeloTTS(t *testing.T) {
	cfg := melotts.Config{
		OnnxRuntimeLibPath: "../lib/onnxruntime.dll",
		ModelPath:          "../melo_weights/model.onnx",
		TokenPath:          "../melo_weights/tokens.txt",
		LexiconPath:        "../melo_weights/lexicon.txt",
	}

	ttsEngine, err := melotts.NewEngine(cfg)
	if err != nil {
		t.Fatalf("创建引擎失败: %v", err)
	}
	defer ttsEngine.Destroy()

	text := "2019年12月30日，中国人口突破14亿人。联系电话: 13800138000。"
	wavData, err := ttsEngine.SynthesizeToWav(text, 1.0)
	if err != nil {
		t.Fatalf("合成失败: %v", err)
	}

	outputPath := "output.wav"
	err = fileutil.FileSave(outputPath, wavData)
	if err != nil {
		t.Fatalf("保存 WAV 失败: %v", err)
	}
}
