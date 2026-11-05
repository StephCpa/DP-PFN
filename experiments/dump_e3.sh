#!/bin/bash
grep -n "Teacher error" -A 30 /tmp/dp-pfn/logs/e3.log | head -40
