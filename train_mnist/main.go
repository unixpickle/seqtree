package main

import (
	"image"
	"image/color"
	"image/png"
	"log"
	"math/rand"
	"os"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/mnist"

	"github.com/unixpickle/seqtree"
)

const (
	Batch     = 10
	ImageSize = 28
	Depth     = 3
	Step      = 0.5

	WarmupStep  = 5.0
	WarmupSteps = 10
)

func main() {
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
	model := &seqtree.Model{BaseFeatures: 256}

	for i := 0; true; i++ {
		seqs := SampleSequences(dataset, model, Batch)
		model.EvaluateAll(seqs)

		var loss float32
		for _, seq := range seqs {
			loss += seq.PropagateLoss()
		}

		tree := seqtree.BuildTree(seqtree.AllTimesteps(seqs...), Depth,
			model.NumFeatures(), horizons)
		if i < WarmupSteps {
			model.Add(tree, WarmupStep)
		} else {
			model.Add(tree, Step)
		}

		log.Printf("step %d: loss=%f", i, loss/(Batch*ImageSize*ImageSize))
		if i%10 == 0 {
			GenerateSequence(model)
		}
	}
}

func SampleSequences(ds mnist.DataSet, m *seqtree.Model, count int) []*seqtree.Timestep {
	var res []*seqtree.Timestep
	for i := 0; i < count; i++ {
		sample := ds.Samples[rand.Intn(len(ds.Samples))]
		intSeq := make([]int, len(sample.Intensities))
		for i, intensity := range sample.Intensities {
			if intensity > 0.5 {
				intSeq[i] = 1
			}
		}
		seq := seqtree.MakeOneHotSequence(intSeq, 2, m.NumFeatures())
		res = append(res, seq)
	}
	return res
}

func GenerateSequence(m *seqtree.Model) {
	img := image.NewGray(image.Rect(0, 0, ImageSize*4, ImageSize*4))
	for row := 0; row < 4; row++ {
		for col := 0; col < 4; col++ {
			seq := &seqtree.Timestep{
				Output:   make([]float32, 2),
				Features: make([]bool, m.NumFeatures()),
			}
			for i := 0; i < ImageSize; i++ {
				for j := 0; j < ImageSize; j++ {
					m.Evaluate(seq)
					num := seqtree.SampleSoftmax(seq.Output)
					if num == 1 {
						img.SetGray(row*ImageSize+j, col*ImageSize+i, color.Gray{Y: 255})
					}
					seq.Next = &seqtree.Timestep{
						Prev:     seq,
						Output:   make([]float32, 2),
						Features: make([]bool, m.NumFeatures()),
					}
					seq.Next.Features[num] = true
					seq = seq.Next
				}
			}
		}
	}
	w, err := os.Create("sample.png")
	essentials.Must(err)
	defer w.Close()
	essentials.Must(png.Encode(w, img))
}
