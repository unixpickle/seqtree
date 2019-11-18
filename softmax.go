package seqtree

import (
	"math"
	"math/rand"

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
	var targetSum float32
	for _, x := range targets {
		targetSum += x
	}

	maxOutput := outputs[0]
	for _, x := range outputs[1:] {
		if x > maxOutput {
			maxOutput = x
		}
	}

	grad := make([]float32, len(outputs))
	for i, x := range outputs {
		grad[i] = float32(math.Exp(float64(x - maxOutput)))
	}
	var gradSum float32
	for _, x := range grad {
		gradSum += x
	}
	div := 1.0 / gradSum
	for i, x := range grad {
		grad[i] = targetSum*x*div - targets[i]
	}
	return grad
}

// SoftmaxLossKL computes
//
//     KL(softmax(outputs+deltas*stepSize)||softmax(outputs))
//
// This can be used to bound the KL of updates.
func SoftmaxLossKL(outputs, deltas []float32, stepSize float32) float32 {
	oldLogProbs := anyvec32.MakeVectorData(outputs)
	newLogProbs := anyvec32.MakeVectorData(deltas)
	newLogProbs.Scale(stepSize)
	newLogProbs.Add(oldLogProbs)
	anyvec.LogSoftmax(oldLogProbs, len(outputs))
	anyvec.LogSoftmax(newLogProbs, len(outputs))
	diff := newLogProbs.Copy()
	diff.Sub(oldLogProbs)
	anyvec.Exp(newLogProbs)
	return newLogProbs.Dot(diff).(float32)
}

// SoftmaxLossDelta computes how much the loss changes due
// to adding deltas*stepSize to outputs.
func SoftmaxLossDelta(outputs, targets, deltas []float32, stepSize float32) float32 {
	targetProbs := anyvec32.MakeVectorData(targets)
	oldLogProbs := anyvec32.MakeVectorData(outputs)
	newLogProbs := anyvec32.MakeVectorData(deltas)
	newLogProbs.Scale(stepSize)
	newLogProbs.Add(oldLogProbs)
	anyvec.LogSoftmax(oldLogProbs, len(outputs))
	anyvec.LogSoftmax(newLogProbs, len(outputs))
	diff := oldLogProbs.Copy()
	diff.Sub(newLogProbs)
	return diff.Dot(targetProbs).(float32)
}
