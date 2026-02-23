package pipertts

const (
	channels      = 1
	bitsPerSample = 16
)

// PiperConfig 对应 .onnx.json 配置文件
type PiperConfig struct {
	Audio struct {
		SampleRate int    `json:"sample_rate"`
		Quality    string `json:"quality"`
	} `json:"audio"`
	Espeak struct {
		Voice string `json:"voice"`
	} `json:"espeak"`
	PhonemeType string `json:"phoneme_type"`
	NumSymbols  int    `json:"num_symbols"`
	NumSpeakers int    `json:"num_speakers"`
	Inference   struct {
		NoiseScale  float32 `json:"noise_scale"`
		LengthScale float32 `json:"length_scale"`
		NoiseW      float32 `json:"noise_w"`
	} `json:"inference"`
	PhonemeIDMap map[string][]int64 `json:"phoneme_id_map"`
}

type Config struct {
	OnnxRuntimeLibPath string
	ModelPath          string
	ConfigPath         string
}
