package melotts

import (
	"fmt"
	"strings"
)

// isAllDigits reports whether s is non-empty and consists entirely of ASCII digits.
func isAllDigits(s string) bool {
	if s == "" {
		return false
	}
	for _, ch := range s {
		if ch < '0' || ch > '9' {
			return false
		}
	}
	return true
}

// expandNumber converts a numeric string to its spoken word form in the given language.
// For ZH the caller handles it via convertutil.TextToChinese upstream, so ZH is a no-op here.
func expandNumber(numStr string, lang string) string {
	// Parse the number; fall back to digit-by-digit on overflow.
	n := int64(0)
	overflow := false
	for _, ch := range numStr {
		d := int64(ch - '0')
		if n > (1<<62)/10 {
			overflow = true
			break
		}
		n = n*10 + d
	}

	switch lang {
	case "en", "en_v2", "en_newest":
		if overflow {
			return digitByDigit(numStr, enDigitWords)
		}
		return numberToWordsEN(n)
	case "zh":
		// Handled upstream by convertutil.TextToChinese
		return numStr
	default:
		// For ko, ja, fr, es — expand digit by digit using language-specific single-digit words.
		words := digitWordsFor(lang)
		if words == nil {
			return numStr
		}
		return digitByDigit(numStr, *words)
	}
}

// digitByDigit converts each character in numStr to the corresponding word from the table.
func digitByDigit(numStr string, words [10]string) string {
	parts := make([]string, 0, len(numStr))
	for _, ch := range numStr {
		if ch >= '0' && ch <= '9' {
			parts = append(parts, words[ch-'0'])
		}
	}
	return strings.Join(parts, " ")
}

// digitWordsFor returns the single-digit word table for a language, or nil if unsupported.
func digitWordsFor(lang string) *[10]string {
	switch lang {
	case "ko":
		// Sino-Korean numerals (used in most counting contexts; all common words in KR lexicon)
		return &[10]string{"영", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"}
	case "ja":
		// Japanese numerals in kanji (present in JP lexicon via MeCab)
		return &[10]string{"零", "一", "二", "三", "四", "五", "六", "七", "八", "九"}
	case "fr":
		// French digit words (0-9 already have entries in FR lexicon, but provide
		// the word form as well so the word lookup path also works)
		return &[10]string{"zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf"}
	case "es":
		return &[10]string{"cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"}
	}
	return nil
}

// ── English number-to-words ───────────────────────────────────────────────────

var enDigitWords = [10]string{
	"zero", "one", "two", "three", "four",
	"five", "six", "seven", "eight", "nine",
}

var enOnes = []string{
	"zero", "one", "two", "three", "four",
	"five", "six", "seven", "eight", "nine",
	"ten", "eleven", "twelve", "thirteen", "fourteen",
	"fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
}

var enTens = []string{
	"", "", "twenty", "thirty", "forty",
	"fifty", "sixty", "seventy", "eighty", "ninety",
}

// numberToWordsEN converts a non-negative integer to English words.
// Handles 0 – 999,999,999,999 (up to hundreds of billions).
func numberToWordsEN(n int64) string {
	if n == 0 {
		return "zero"
	}
	return strings.TrimSpace(enChunk(n))
}

func enChunk(n int64) string {
	switch {
	case n < 0:
		return "minus " + enChunk(-n)
	case n < 20:
		return enOnes[n]
	case n < 100:
		rest := ""
		if n%10 != 0 {
			rest = " " + enOnes[n%10]
		}
		return enTens[n/10] + rest
	case n < 1000:
		rest := ""
		if n%100 != 0 {
			rest = " " + enChunk(n%100)
		}
		return fmt.Sprintf("%s hundred%s", enOnes[n/100], rest)
	case n < 1_000_000:
		rest := ""
		if n%1000 != 0 {
			rest = " " + enChunk(n%1000)
		}
		return fmt.Sprintf("%s thousand%s", enChunk(n/1000), rest)
	case n < 1_000_000_000:
		rest := ""
		if n%1_000_000 != 0 {
			rest = " " + enChunk(n%1_000_000)
		}
		return fmt.Sprintf("%s million%s", enChunk(n/1_000_000), rest)
	default:
		rest := ""
		if n%1_000_000_000 != 0 {
			rest = " " + enChunk(n%1_000_000_000)
		}
		return fmt.Sprintf("%s billion%s", enChunk(n/1_000_000_000), rest)
	}
}
