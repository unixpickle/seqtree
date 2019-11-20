package seqtree

import (
	"math"
	"math/rand"
	"testing"
)

func TestSoftmaxLoss(t *testing.T) {
	for i := 0; i < 100; i++ {
		vec := make([]float32, i+3)
		target := make([]float32, len(vec))
		for i := range vec {
			vec[i] = float32(rand.NormFloat64())
			target[i] = float32(rand.NormFloat64())
		}
		actual := SoftmaxLoss(vec, target)
		expected := float32(0)
		log := logSoftmax(vec)
		for i, l := range log {
			expected -= l * target[i]
		}
		if math.Abs(float64(actual-expected)) > 1e-4 {
			t.Errorf("expected %f but got %f", expected, actual)
		}
	}
}

func BenchmarkSoftmaxLossGrad(b *testing.B) {
	for i := 0; i < b.N; i++ {
		SoftmaxLossGrad([]float32{1.5, -0.3}, []float32{1.0, 0})
	}
}

func BenchmarkSoftmaxLossDelta(b *testing.B) {
	for i := 0; i < b.N; i++ {
		SoftmaxLossDelta([]float32{1.5, -0.3}, []float32{1.0, 0}, []float32{0.5, -0.2}, 0.1)
	}
}

func BenchmarkSoftmaxLossKL(b *testing.B) {
	for i := 0; i < b.N; i++ {
		SoftmaxLossKL([]float32{1.5, -0.3}, []float32{0.5, -0.2}, 0.1)
	}
}
