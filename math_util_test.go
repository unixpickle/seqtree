package seqtree

import (
	"math"
	"testing"
)

func TestPolynomialLogSigmoid(t *testing.T) {
	centers := []float32{-30, -20, -10, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 10, 20, 30}
	for _, center := range centers {
		p := newPolynomialLogSigmoid(center)
		for x := float32(-1.0); x <= 1.0; x += 0.1 {
			actual := p.Evaluate(x)
			expected := exactLogSigmoid(center + x)
			if math.Abs(float64(actual-expected)) > 1e-5 {
				t.Errorf("log(sigmoid(%f+%f)) should be %f but got %f",
					center, x, expected, actual)
			}
		}
	}
}

func exactLogSigmoid(x float32) float32 {
	if x < -20 {
		return x
	} else if x > 20 {
		return float32(math.Exp(float64(-x)))
	}
	return float32(math.Log(1 / (1 + math.Exp(float64(-x)))))
}
