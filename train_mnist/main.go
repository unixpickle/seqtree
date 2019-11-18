package main

import (
	"image"
	"image/color"
	"image/png"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/mnist"

	"github.com/unixpickle/seqtree"
)

const (
	Batch           = 2000
	ImageSize       = 28
	Depth           = 4
	MaxKL           = 0.01
	MaxStep         = 10.0
	MinSplitSamples = 10

	// Split with a small subset of the entire batch.
	MaxSplitSamples = 10 * ImageSize * ImageSize
)

func main() {
	rand.Seed(time.Now().UnixNano())
	horizons := []int{}
	for i := -4; i <= 4; i++ {
		for j := 0; j <= 4; j++ {
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
		MinSplitSamples: MinSplitSamples,
		MaxSplitSamples: MaxSplitSamples,
	}

	for i := 0; true; i++ {
		seqs := SampleSequences(dataset, model, Batch)
		model.EvaluateAll(seqs)

		var loss float32
		for _, seq := range seqs {
			loss += seq.PropagateLoss()
		}

		builder.ExtraFeatures = model.ExtraFeatures
		tree := builder.Build(seqtree.AllTimesteps(seqs...))

		// Bound KL on a different batch.
		seqs = SampleSequences(dataset, model, Batch)
		model.EvaluateAll(seqs)

		stepSize := seqtree.BoundedStep(seqtree.AllTimesteps(seqs...), tree, MaxKL, MaxStep)
		if stepSize > 0 {
			model.Add(tree, stepSize)
		}

		log.Printf("step %d: loss=%f step_size=%f", i, loss/Batch, stepSize)
		if i%10 == 0 {
			GenerateSequence(model)
		}
		model.Save("model.json")
	}
}

func SampleSequences(ds mnist.DataSet, m *seqtree.Model, count int) []seqtree.Sequence {
	var res []seqtree.Sequence
	for i := 0; i < count; i++ {
		sample := ds.Samples[rand.Intn(len(ds.Samples))]
		seq := seqtree.Sequence{}
		prev := -1
		for i, intensity := range sample.Intensities {
			x := i % ImageSize
			y := i / ImageSize
			ts := &seqtree.Timestep{
				Output:   make([]float32, 2),
				Features: make([]bool, m.NumFeatures()),
				Target:   make([]float32, 2),
			}
			if prev != -1 {
				ts.Features[prev] = true
			}
			ts.Features[2+x] = true
			ts.Features[2+ImageSize+y] = true
			if intensity > 0.5 {
				prev = 1
			} else {
				prev = 0
			}
			ts.Target[prev] = 1.0
			seq = append(seq, ts)
		}
		res = append(res, seq)
	}
	return res
}

func GenerateSequence(m *seqtree.Model) {
	img := image.NewGray(image.Rect(0, 0, ImageSize*4, ImageSize*4))
	for row := 0; row < 4; row++ {
		for col := 0; col < 4; col++ {
			seq := seqtree.Sequence{
				&seqtree.Timestep{
					Output:   make([]float32, 2),
					Features: make([]bool, m.NumFeatures()),
				},
			}
			for i := 0; i < ImageSize; i++ {
				for j := 0; j < ImageSize; j++ {
					ts := seq[len(seq)-1]
					ts.Features[2+j] = true
					ts.Features[2+ImageSize+i] = true
					m.EvaluateAt(seq, len(seq)-1)
					num := seqtree.SampleSoftmax(ts.Output)
					if num == 1 {
						img.SetGray(row*ImageSize+j, col*ImageSize+i, color.Gray{Y: 255})
					}
					ts = &seqtree.Timestep{
						Output:   make([]float32, 2),
						Features: make([]bool, m.NumFeatures()),
					}
					ts.Features[num] = true
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
