#!/bin/sh
gunicorn  run:app -w 2 --threads 1 -b 0.0.0.0:80