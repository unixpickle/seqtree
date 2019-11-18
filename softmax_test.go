package seqtree

import "testing"

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
