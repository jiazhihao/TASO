#!/bin/bash
g++ generator.cc rules.pb.cc -o generator -I/usr/local/include -L/usr/local/lib -lprotobuf -std=c++11 -pthread -O2
