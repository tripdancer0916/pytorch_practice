# !/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import unicode_literals
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


