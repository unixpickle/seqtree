package seqtree

import (
	"math"
	"math/rand"
	"strconv"
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

func TestHessianMatrixSoftmax(t *testing.T) {
	for _, dim := range []int{2, 10} {
		t.Run("Dim"+strconv.Itoa(dim), func(t *testing.T) {
			for i := 0; i < 10; i++ {
				logits := make([]float32, dim)
				targets := make([]float32, dim)
				for i := range logits {
					logits[i] = float32(rand.NormFloat64())
					targets[i] = float32(rand.NormFloat64())
				}
				actual := newHessianMatrixSoftmax(logits, targets)
				expected := approxHessianMatrixSoftmax(logits, targets)
				for i, x := range expected.Values {
					a := actual.Values[i]
					if math.IsNaN(float64(a)) {
						t.Fatal("got NaN")
					} else if math.Abs(float64(a-x)) > 1e-3 {
						t.Errorf("expected second derivative %f but got %f (idx %d)", x, a, i)
					}
				}
			}
		})
	}
}

func approxHessianMatrixSoftmax(logits, targets []float32) *hessianMatrix {
	const epsilon = 1e-3
	res := &hessianMatrix{
		Dim:    len(logits),
		Values: make([]float32, len(logits)*len(logits)),
	}
	for i := range logits {
		l1 := append([]float32{}, logits...)
		l1[i] -= epsilon
		g1 := SoftmaxLossGrad(l1, targets)
		l1[i] += epsilon * 2
		g2 := SoftmaxLossGrad(l1, targets)
		for j := range logits {
			res.Values[i+j*len(logits)] = (g2[j] - g1[j]) / (epsilon * 2)
		}
	}
	return res
}
