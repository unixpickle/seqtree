package seqtree

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec64"
)

// SoftmaxLoss computes the loss function given output
// logits and target probabilities.
func SoftmaxLoss(outputs, targets []float64) float64 {
	probs := anyvec64.MakeVectorData(targets)
	logProbs := anyvec64.MakeVectorData(outputs)
	anyvec.LogSoftmax(logProbs, len(outputs))
	return -logProbs.Dot(probs).(float64)
}

// SoftmaxLossGrad computes the gradient of SoftmaxLoss
// with respect to the outputs.
func SoftmaxLossGrad(outputs, targets []float64) []float64 {
	probs := anydiff.NewConst(anyvec64.MakeVectorData(targets))
	logits := anydiff.NewVar(anyvec64.MakeVectorData(outputs))
	prob := anydiff.Dot(anydiff.LogSoftmax(logits, len(outputs)), probs)
	prob = anydiff.Scale(prob, -1.0)
	grad := anydiff.NewGrad(logits)
	prob.Propagate(anyvec64.MakeVectorData([]float64{1.0}), grad)
	return grad[logits].Data().([]float64)
}
