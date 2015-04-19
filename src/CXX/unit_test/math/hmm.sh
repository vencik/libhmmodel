#!/bin/sh

path="Healthy Healthy Fever"
probability=0.01512

./hmm | grep "^Path (probability $probability): $path" || exit 1
