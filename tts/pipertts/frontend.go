package pipertts

import (
	"fmt"
	"github.com/mozillazg/go-pinyin"
	"regexp"
	"strings"
)

// textToIds 将中文文本转换为 Piper 识别的 ID 序列
func (e *Engine) textToIds(text string) []int64 {
	phonemes := e.toPhonemes(text)
	return e.phonemesToIds(phonemes)
}

// toPhonemes 将中文文本转换为 Piper 识别的音素序列
func (e *Engine) toPhonemes(text string) string {
	// 设置拼音转换参数：使用声调模式 (如: hǎo -> hao3)
	args := pinyin.NewArgs()
	args.Style = pinyin.Tone3
	args.Fallback = func(r rune, a pinyin.Args) []string {
		return []string{string(r)}
	}

	pys := pinyin.Pinyin(text, args)

	var result []string
	// 正则用于识别带声调的拼音 (如 hao3) 和 不带声调的拼音 (如 le)
	pyWithToneReg := regexp.MustCompile(`^([a-z]+)([1-5])$`)
	pureAlphaReg := regexp.MustCompile(`^[a-z]+$`)

	for _, p := range pys {
		token := p[0]

		var fullPy string
		var tone string

		if matches := pyWithToneReg.FindStringSubmatch(token); len(matches) == 3 {
			fullPy = matches[1]
			tone = matches[2]
		} else if pureAlphaReg.MatchString(token) {
			// 处理轻声情况：如 "le", "de" 补全为 5 声
			fullPy = token
			tone = "5"
		}

		if fullPy != "" {
			// 拆分声母和韵母
			initial := getInitial(fullPy)
			final := strings.TrimPrefix(fullPy, initial)

			result = append(result, "_")
			if initial != "" {
				result = append(result, initial)
			}
			if final != "" {
				// 将韵母和声调分开作为独立音素
				result = append(result, final, tone)
			}
		} else {
			// 根据提供的 phoneme_id_map 处理标点符号
			switch token {
			case ",", "，", "、", "—", "…", ":", "：", ";", "；":
				result = append(result, ",") // 对应 ID 72
			case ".", "。", "?", "？", "!", "！":
				result = append(result, ".") // 对应 ID 69 或 71
			case " ":
				result = append(result, " ")
			default:
				result = append(result, token)
			}
		}
	}

	// 保持末尾有一个停顿感
	if len(result) > 0 && result[len(result)-1] != "." {
		result = append(result, ".")
	}

	return strings.Join(result, " ")
}

// phonemesToIds 将以空格分隔的音素字符串转为 ID 序列
func (e *Engine) phonemesToIds(phonemesText string) []int64 {
	ids := make([]int64, 0)

	tokens := strings.Split(strings.TrimSpace(phonemesText), " ")
	for _, token := range tokens {
		if token == "" {
			continue
		}

		// 查找 JSON 中的 phoneme_id_map
		if mappedIDs, ok := e.piperConfig.PhonemeIDMap[token]; ok {
			ids = append(ids, mappedIDs...)
		} else {
			fmt.Printf("[WARN] 未找到音素映射: %s\n", token)
		}
	}

	return ids
}
