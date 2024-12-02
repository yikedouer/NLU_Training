#!/usr/bin/env bash
keyword=$1
ps -aux | grep -ie ${keyword} | grep -v grep | awk '{print $2}' | xargs kill -9
