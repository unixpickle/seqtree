package main

import "github.com/unixpickle/mnist"

const (
	ImageSize      = 28
	WindowSize     = 10
	SequenceLength = (WindowSize*2 + 2) * WindowSize
)

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

func (b BoolImg) Window(x, y int) []bool {
	result := make([]bool, 0, SequenceLength)
	for j := WindowSize; j > 0; j-- {
		for i := -WindowSize; i <= WindowSize; i++ {
			result = append(result, b.At(x+i, y-j))
		}
	}
	for i := -WindowSize; i < 0; i++ {
		result = append(result, b.At(x+i, y))
	}
	return result
}
