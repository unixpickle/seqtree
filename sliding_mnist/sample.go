package main

import "github.com/unixpickle/mnist"

const (
	ImageSize      = 28
	WindowSize     = 10
	SequenceLength = (WindowSize*2 + 2) * WindowSize
)

type FeatureMap struct {
	Img BoolImg
	X   int
	Y   int
}

func (f *FeatureMap) Len() int {
	return SequenceLength + ImageSize*2
}

func (f *FeatureMap) Get(i int) bool {
	if i < SequenceLength {
		return f.Img.GetWindow(f.X, f.Y, i)
	} else if i < SequenceLength+ImageSize {
		return f.X == i-SequenceLength
	} else {
		return f.Y == i-(SequenceLength+ImageSize)
	}
}

func (f *FeatureMap) Set(i int, b bool) {
	panic("feature map is immutable")
}

type BoolImg []bool

func NewBoolImg() BoolImg {
	return make(BoolImg, ImageSize*ImageSize)
}

func NewBoolImgSample(sample mnist.Sample) BoolImg {
	res := NewBoolImg()
	for i, x := range sample.Intensities {
		res[i] = x >= 0.5
	}
	return res
}

func (b BoolImg) At(x, y int) bool {
	if x < 0 || x >= ImageSize || y < 0 || y >= ImageSize {
		return false
	}
	return b[x+y*ImageSize]
}

func (b BoolImg) Set(x, y int, v bool) {
	b[x+y*ImageSize] = v
}

func (b BoolImg) GetWindow(x, y int, idx int) bool {
	rowIdx := idx / (WindowSize*2 + 1)
	colIdx := idx % (WindowSize*2 + 1)
	return b.At(x-WindowSize+colIdx, y-WindowSize+rowIdx)
}
