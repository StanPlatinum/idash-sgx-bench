all: imagenet_parser imagenet_parser-m3

imagenet_parser: imagenet_parser.cpp
	g++ -O2 -std=c++11 imagenet_parser.cpp -o imagenet_parser

imagenet_parser-m3: imagenet_parser-m3.cpp
	g++ -O2 -std=c++11 imagenet_parser-m3.cpp -o imagenet_parser-m3

clean:
	rm -f imagenet_parser
	rm -f imagenet_parser-m3
