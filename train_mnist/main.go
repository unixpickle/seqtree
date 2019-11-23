package main

import (
	"image"
	"image/color"
	"image/png"
	"log"
	"math/rand"
	"os"
	"runtime"
	"sync"
	"time"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/mnist"

	"github.com/unixpickle/seqtree"
)

const (
	ImageSize                = 28
	HorizontalReceptiveField = 5
	VerticalReceptiveField   = 9

	Batch    = 1000
	Depth    = 4
	MaxUnion = 3
	MaxStep  = 20.0

	MinSplitSamplesMin = 100
	MinSplitSamplesMax = 1000

	// Split with a small subset of the entire batch.
	MaxSplitSamples = 10 * ImageSize * ImageSize
	CandidateSplits = 20
)

func main() {
	rand.Seed(time.Now().UnixNano())
	horizons := []int{}
	for i := -HorizontalReceptiveField; i < HorizontalReceptiveField; i++ {
		for j := 0; j < VerticalReceptiveField; j++ {
			if j == 0 && i < 0 {
				continue
			}
			horizons = append(horizons, i+j*ImageSize)
		}
	}
	dataset := mnist.LoadTrainingDataSet()
	model := &seqtree.Model{BaseFeatures: 2 + ImageSize*2}
	model.Load("model.json")

	builder := seqtree.Builder{
		Depth:           Depth,
		Horizons:        horizons,
		MaxSplitSamples: MaxSplitSamples,
		MaxUnion:        MaxUnion,
		CandidateSplits: CandidateSplits,
	}

	for i := 0; true; i++ {
		seqs := SampleSequences(dataset, model, Batch)
		model.EvaluateAll(seqs)
		seqtree.PropagateLosses(seqs)
		seqtree.PropagateHessians(seqs)

		totalLoss := float32(0)
		for _, seq := range seqs {
			totalLoss += seq.MeanLoss()
		}
		totalLoss /= Batch

		builder.MinSplitSamples = rand.Intn(MinSplitSamplesMax-MinSplitSamplesMin) +
			MinSplitSamplesMin
		builder.ExtraFeatures = model.ExtraFeatures
		tree := builder.Build(seqtree.AllTimesteps(seqs...))

		// Optimize step size on a different batch.
		seqs = SampleSequences(dataset, model, Batch)
		model.EvaluateAll(seqs)

		seqtree.ScaleOptimalStep(seqtree.AllTimesteps(seqs...), tree, MaxStep, 10, 20)
		delta := seqtree.AvgLossDelta(seqtree.AllTimesteps(seqs...), tree, 1.0)
		model.Add(tree, 1.0)

		log.Printf("step %d: loss=%f loss_delta=%f min_leaf=%d",
			i, totalLoss, -delta, builder.MinSplitSamples)

		GenerateSequence(model)
		model.Save("model.json")
	}
}

func SampleSequences(ds mnist.DataSet, m *seqtree.Model, count int) []seqtree.Sequence {
	res := make([]seqtree.Sequence, count)

	var wg sync.WaitGroup
	numProcs := runtime.GOMAXPROCS(0)
	for i := 0; i < numProcs; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			for j := 0; j < count; j++ {
				if j%numProcs != i {
					continue
				}
				sample := ds.Samples[rand.Intn(len(ds.Samples))]
				seq := seqtree.Sequence{}
				prev := -1
				for i, intensity := range sample.Intensities {
					x := i % ImageSize
					y := i / ImageSize
					ts := &seqtree.Timestep{
						Output:   make([]float32, 2),
						Features: seqtree.NewBitmap(m.NumFeatures()),
						Target:   make([]float32, 2),
					}
					if prev != -1 {
						ts.Features.Set(prev, true)
					}
					ts.Features.Set(2+x, true)
					ts.Features.Set(2+ImageSize+y, true)
					if intensity > 0.5 {
						prev = 1
					} else {
						prev = 0
					}
					ts.Target[prev] = 1.0
					seq = append(seq, ts)
				}
				res[j] = seq
			}
		}(i)
	}
	wg.Wait()
	return res
}

func GenerateSequence(m *seqtree.Model) {
	img := image.NewGray(image.Rect(0, 0, ImageSize*4, ImageSize*4))
	for row := 0; row < 4; row++ {
		for col := 0; col < 4; col++ {
			seq := seqtree.Sequence{
				&seqtree.Timestep{
					Output:   make([]float32, 2),
					Features: seqtree.NewBitmap(m.NumFeatures()),
				},
			}
			for i := 0; i < ImageSize; i++ {
				for j := 0; j < ImageSize; j++ {
					ts := seq[len(seq)-1]
					ts.Features.Set(2+j, true)
					ts.Features.Set(2+ImageSize+i, true)
					m.EvaluateAt(seq, len(seq)-1)
					num := seqtree.SampleSoftmax(ts.Output)
					if num == 1 {
						img.SetGray(row*ImageSize+j, col*ImageSize+i, color.Gray{Y: 255})
					}
					ts = &seqtree.Timestep{
						Output:   make([]float32, 2),
						Features: seqtree.NewBitmap(m.NumFeatures()),
					}
					ts.Features.Set(num, true)
					seq = append(seq, ts)
				}
			}
		}
	}
	w, err := os.Create("sample.png")
	essentials.Must(err)
	defer w.Close()
	essentials.Must(png.Encode(w, img))
}
