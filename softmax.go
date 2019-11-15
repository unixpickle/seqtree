package seqtree

import (
	"math/rand"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
)

// SampleSoftmax samples a softmax distribution from
// logits.
func SampleSoftmax(outputs []float32) int {
	probs := anyvec32.MakeVectorData(outputs)
	anyvec.LogSoftmax(probs, len(outputs))
	anyvec.Exp(probs)
	vec := probs.Data().([]float32)
	p := rand.Float32()
	for i, x := range vec {
		p -= x
		if p <= 0 {
			return i
		}
	}
	return len(vec) - 1
}

// SoftmaxLoss computes the loss function given output
// logits and target probabilities.
func SoftmaxLoss(outputs, targets []float32) float32 {
	probs := anyvec32.MakeVectorData(targets)
	logProbs := anyvec32.MakeVectorData(outputs)
	anyvec.LogSoftmax(logProbs, len(outputs))
	return -logProbs.Dot(probs).(float32)
}

// SoftmaxLossGrad computes the gradient of SoftmaxLoss
// with respect to the outputs.
func SoftmaxLossGrad(outputs, targets []float32) []float32 {
	probs := anydiff.NewConst(anyvec32.MakeVectorData(targets))
	logits := anydiff.NewVar(anyvec32.MakeVectorData(outputs))
	prob := anydiff.Dot(anydiff.LogSoftmax(logits, len(outputs)), probs)
	prob = anydiff.Scale(prob, float32(-1.0))
	grad := anydiff.NewGrad(logits)
	prob.Propagate(anyvec32.MakeVectorData([]float32{1.0}), grad)
	return grad[logits].Data().([]float32)
}
