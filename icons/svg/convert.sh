#!/bin/sh

mogrify -background none -resize 80x80 -format png "$@"
