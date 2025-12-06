package melotts

import (
	"fmt"
	"strings"
)

// textToIds 将标准化后的文本转换为 Token ID 和 Tone ID 序列
func (e *Engine) textToIds(text string) ([]int64, []int64, error) {
	var ids []int64
	var tones []int64

	// 使用智能分词将文本切分为 单词/汉字/标点
	segments := smartSegment(text)

	for _, word := range segments {
		if strings.TrimSpace(word) == "" {
			continue
		}
		lowerWord := strings.ToLower(word)

		// 查词典 (优先全匹配)
		if e.appendIdsFromLexicon(lowerWord, &ids, &tones) {
			continue
		}

		// 查 Token
		if id, ok := e.tokenMap[word]; ok {
			e.appendToken(id, 0, &ids, &tones)
			continue
		}

		// OOV 逐字符降级处理 (Character Fallback)
		runes := []rune(word)
		if len(runes) > 1 {
			var subIds, subTones []int64

			for _, r := range runes {
				charStr := string(r)
				lowerChar := strings.ToLower(charStr)

				// 尝试单字符查词典 (如 'a' -> [phone...])
				if e.appendIdsFromLexicon(lowerChar, &subIds, &subTones) {
					continue
				}

				// 尝试单字符直接查 Token
				if id, ok := e.tokenMap[charStr]; ok {
					e.appendToken(id, 0, &subIds, &subTones)
					continue
				}

				// 单字符也无法处理
				fmt.Printf("[WARN] OOV 字符丢失: %s (in %s)\n", charStr, word)
			}

			if len(subIds) > 0 {
				ids = append(ids, subIds...)
				tones = append(tones, subTones...)
				continue
			}
		}

		// 完全无法处理
		fmt.Printf("[WARN] 跳过完全未识别字符/单词: %s\n", word)
	}

	// 结尾 Pad
	e.appendToken(0, 0, &ids, &tones)

	if len(ids) <= 1 {
		return nil, nil, fmt.Errorf("生成的 Token 序列为空")
	}
	return ids, tones, nil
}

// appendIdsFromLexicon 尝试从词典查找并追加 IDs，返回是否成功
func (e *Engine) appendIdsFromLexicon(key string, ids *[]int64, tones *[]int64) bool {
	item, ok := e.lexicon[key]
	if !ok {
		return false
	}

	for i, phone := range item.Phones {
		id, ok := e.tokenMap[phone]
		if !ok {
			fmt.Printf("[ERROR] Lexicon 含未知音素: %s (key: %s)\n", phone, key)
			continue
		}
		tVal := item.Tones[i]
		e.appendToken(id, tVal, ids, tones)
	}
	return true
}

// appendToken 辅助函数：统一追加格式 [0, id]
func (e *Engine) appendToken(id int64, tone int64, ids *[]int64, tones *[]int64) {
	*ids = append(*ids, 0, id)
	*tones = append(*tones, 0, tone)
}

// smartSegment 简单的分词逻辑: 区分汉字/符号与英文/数字单词
func smartSegment(text string) []string {
	var segments []string
	var buffer strings.Builder
	runes := []rune(text)

	for i := 0; i < len(runes); i++ {
		r := runes[i]
		// 英文、数字、单引号作为单词的一部分连续处理
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '\'' {
			buffer.WriteRune(r)
		} else {
			// 遇到非英文/数字，先结算之前的 buffer
			if buffer.Len() > 0 {
				segments = append(segments, buffer.String())
				buffer.Reset()
			}
			// 当前字符作为独立段 (汉字、标点)
			segments = append(segments, string(r))
		}
	}
	// 结算剩余 buffer
	if buffer.Len() > 0 {
		segments = append(segments, buffer.String())
	}
	return segments
}
