package whisper

import (
	"github.com/getcharzp/go-speech"
)

const (
	// LangEn 英语
	LangEn = "en"
	// LangZh 中文
	LangZh = "zh"
	// LangDe 德语
	LangDe = "de"
	// LangEs 西班牙语
	LangEs = "es"
	// LangRu 俄语
	LangRu = "ru"
	// LangKo 韩语
	LangKo = "ko"
	// LangFr 法语
	LangFr = "fr"
	// LangJa 日语
	LangJa = "ja"
	// LangPt 葡萄牙语
	LangPt = "pt"
	// LangTr 土耳其语
	LangTr = "tr"
	// LangPl 波兰语
	LangPl = "pl"
	// LangCa 加泰罗尼亚语
	LangCa = "ca"
	// LangNl 荷兰语
	LangNl = "nl"
	// LangAr 阿拉伯语
	LangAr = "ar"
	// LangSv 瑞典语
	LangSv = "sv"
	// LangIt 意大利语
	LangIt = "it"
	// LangId 印尼语
	LangId = "id"
	// LangHi 印地语
	LangHi = "hi"
	// LangFi 芬兰语
	LangFi = "fi"
	// LangVi 越南语
	LangVi = "vi"
	// LangHe 希伯来语
	LangHe = "he"
	// LangUk 乌克兰语
	LangUk = "uk"
	// LangEl 希腊语
	LangEl = "el"
	// LangMs 马来语
	LangMs = "ms"
	// LangCs 捷克语
	LangCs = "cs"
	// LangRo 罗马尼亚语
	LangRo = "ro"
	// LangDa 丹麦语
	LangDa = "da"
	// LangHu 匈牙利语
	LangHu = "hu"
	// LangTa 泰米尔语
	LangTa = "ta"
	// LangNo 挪威语
	LangNo = "no"
	// LangTh 泰语
	LangTh = "th"
	// LangUr 乌尔都语
	LangUr = "ur"
	// LangHr 克罗地亚语
	LangHr = "hr"
	// LangBg 保加利亚语
	LangBg = "bg"
	// LangLt 立陶宛语
	LangLt = "lt"
	// LangLa 拉丁语
	LangLa = "la"
	// LangMi 毛利语
	LangMi = "mi"
	// LangMl 马拉雅拉姆语
	LangMl = "ml"
	// LangCy 威尔士语
	LangCy = "cy"
	// LangSk 斯洛伐克语
	LangSk = "sk"
	// LangTe 泰卢固语
	LangTe = "te"
	// LangFa 波斯语
	LangFa = "fa"
	// LangLv 拉脱维亚语
	LangLv = "lv"
	// LangBn 孟加拉语
	LangBn = "bn"
	// LangSr 塞尔维亚语
	LangSr = "sr"
	// LangAz 阿塞拜疆语
	LangAz = "az"
	// LangSl 斯洛文尼亚语
	LangSl = "sl"
	// LangKn 卡纳达语
	LangKn = "kn"
	// LangEt 爱沙尼亚语
	LangEt = "et"
	// LangMk 马其顿语
	LangMk = "mk"
	// LangBr 布列塔尼语
	LangBr = "br"
	// LangEu 巴斯克语
	LangEu = "eu"
	// LangIs 冰岛语
	LangIs = "is"
	// LangHy 亚美尼亚语
	LangHy = "hy"
	// LangNe 尼泊尔语
	LangNe = "ne"
	// LangMn 蒙古语
	LangMn = "mn"
	// LangBs 波斯尼亚语
	LangBs = "bs"
	// LangKk 哈萨克语
	LangKk = "kk"
	// LangSq 阿尔巴尼亚语
	LangSq = "sq"
	// LangSw 斯瓦希里语
	LangSw = "sw"
	// LangGl 加利西亚语
	LangGl = "gl"
	// LangMr 马拉地语
	LangMr = "mr"
	// LangPa 旁遮普语
	LangPa = "pa"
	// LangSi 僧伽罗语
	LangSi = "si"
	// LangKm 高棉语
	LangKm = "km"
	// LangSn 绍纳语
	LangSn = "sn"
	// LangYo 约鲁巴语
	LangYo = "yo"
	// LangSo 索马里语
	LangSo = "so"
	// LangAf 南非荷兰语
	LangAf = "af"
	// LangOc 欧西坦语
	LangOc = "oc"
	// LangKa 格鲁吉亚语
	LangKa = "ka"
	// LangBe 白俄罗斯语
	LangBe = "be"
	// LangTg 塔吉克语
	LangTg = "tg"
	// LangSd 信德语
	LangSd = "sd"
	// LangGu 古吉拉特语
	LangGu = "gu"
	// LangAm 阿姆哈拉语
	LangAm = "am"
	// LangYi 意第绪语
	LangYi = "yi"
	// LangLo 老挝语
	LangLo = "lo"
	// LangUz 乌兹别克语
	LangUz = "uz"
	// LangFo 法罗语
	LangFo = "fo"
	// LangHt 海地克里奥尔语
	LangHt = "ht"
	// LangPs 普什图语
	LangPs = "ps"
	// LangTk 土库曼语
	LangTk = "tk"
	// LangNn 新挪威语
	LangNn = "nn"
	// LangMt 马耳他语
	LangMt = "mt"
	// LangSa 梵语
	LangSa = "sa"
	// LangLb 卢森堡语
	LangLb = "lb"
	// LangMy 缅甸语
	LangMy = "my"
	// LangBo 藏语
	LangBo = "bo"
	// LangTl 塔加路语
	LangTl = "tl"
	// LangMg 马达加斯加语
	LangMg = "mg"
	// LangAs 阿萨姆语
	LangAs = "as"
	// LangTt 韦语
	LangTt = "tt"
	// LangHaw 夏威夷语
	LangHaw = "haw"
	// LangLn 林加拉语
	LangLn = "ln"
	// LangHa 豪萨语
	LangHa = "ha"
	// LangBa 巴什基尔语
	LangBa = "ba"
	// LangJw 爪哇语
	LangJw = "jw"
	// LangSu 苏丹语
	LangSu = "su"
)

const (
	// TaskTranscribe 转录
	TaskTranscribe = "transcribe"
	// TaskTranslate 翻译
	TaskTranslate = "translate"
)

// Config Whisper 模型的配置参数
type Config struct {
	// 必填参数
	OnnxRuntimeLibPath string
	DecoderModelPath   string
	EncoderModelPath   string
	TokensPath         string // vocab.json 文件路径
	AddedTokensPath    string // added_tokens.json 文件路径
	ModelLayers        int
	MaxTokens          int

	// 可选参数
	UseCuda    bool // (可选) 是否启用 CUDA
	NumThreads int  // (可选) ONNX 线程数, 默认由CPU核心数决定
}

// DefaultConfig 默认配置
func DefaultConfig() Config {
	return Config{
		OnnxRuntimeLibPath: speech.DefaultLibraryPath(),
		EncoderModelPath:   "./whisper_weights/small_encoder_model.onnx",
		DecoderModelPath:   "./whisper_weights/small_decoder_model_merged.onnx", // 默认使用 small 模型
		TokensPath:         "./whisper_weights/vocab.json",
		AddedTokensPath:    "./whisper_weights/added_tokens.json",
		ModelLayers:        12,
		MaxTokens:          200,
	}
}

// TranscribeOption 转录配置参数
type TranscribeOption struct {
	Language string // 被转录的语言，例如："zh", "en", "ja"
	Task     string // 任务类型，例如："transcribe", "translate"
}
