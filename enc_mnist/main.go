package main

import (
	"log"
	"math/rand"

	"github.com/unixpickle/mnist"
	"github.com/unixpickle/seqtree"
)

const (
	ImageSize       = 28
	Batch           = 1 << 13
	Depth           = 4
	MinSplitSamples = 2
	MaxUnion        = 5
)

func main() {
	dataset := mnist.LoadTrainingDataSet()
	model := &seqtree.Model{BaseFeatures: ImageSize * ImageSize}
	model.Load("model.json")
	builder := seqtree.Builder{
		Heuristic:       seqtree.GradientHeuristic{Loss: seqtree.Sigmoid{}},
		Depth:           Depth,
		MinSplitSamples: MinSplitSamples,
		MaxUnion:        MaxUnion,
		Horizons:        []int{0},
	}

	for {
		samples := GenerateSamples(dataset, Batch)
		model.EvaluateAll(samples)

		loss := float32(0)
		for _, s := range samples {
			loss += s.MeanLoss(seqtree.Sigmoid{})
		}

		if len(model.Trees) != 0 {
			// First step benefits from GradientHeuristic.
			builder.Heuristic = seqtree.PolynomialHeuristic{Loss: seqtree.Sigmoid{}}
		}
		tree := builder.Build(seqtree.TimestepSamples(samples))

		samples = GenerateSamples(dataset, Batch)
		model.EvaluateAll(samples)
		seqtree.ScaleOptimalStep(seqtree.TimestepSamples(samples), tree, seqtree.Sigmoid{},
			20, 1, 30)
		delta := seqtree.AvgLossDelta(seqtree.TimestepSamples(samples), tree, seqtree.Sigmoid{},
			1.0)
		model.Add(tree, 1.0)

		log.Printf("step %d: loss=%f delta=%f", len(model.Trees)-1, loss/Batch, -delta)
		model.Save("model.json")
		SaveReconstructions(dataset, model)
	}
}

func GenerateSamples(ds mnist.DataSet, n int) []seqtree.Sequence {
	var res []seqtree.Sequence
	for i := 0; i < n; i++ {
		sample := ds.Samples[rand.Intn(len(ds.Samples))]
		res = append(res, seqtree.Sequence{sampleToTimestep(sample.Intensities)})
	}
	return res
}

func sampleToTimestep(intensities []float64) *seqtree.Timestep {
	ts := &seqtree.Timestep{
		Features: seqtree.NewBitmap(ImageSize * ImageSize),
		Output:   make([]float32, ImageSize*ImageSize),
		Target:   make([]float32, ImageSize*ImageSize),
	}
	for i, s := range intensities {
		if s >= 0.5 {
			ts.Features.Set(i, true)
			ts.Target[i] = 1.0
		}
	}
	return ts
}

func SaveReconstructions(ds mnist.DataSet, m *seqtree.Model) {
	mnist.SaveReconstructionGrid("recon.png", func(x []float64) []float64 {
		ts := sampleToTimestep(x)
		m.Evaluate(seqtree.Sequence{ts})
		res := make([]float64, len(ts.Output))
		samples := seqtree.Sigmoid{}.Sample(ts.Output)
		for i, x := range samples {
			if x {
				res[i] = 1
			}
		}
		return res
	}, ds, 8, 4)
}
