# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from transformers import StoppingCriteria


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        # Ensure stop tokens are on the same device as input_ids
        device = input_ids.device
        for stop in self.stops:
            # Move stop to the same device as input_ids if it's a tensor
            if isinstance(stop, torch.Tensor):
                stop = stop.to(device)
            else:
                # If stop is a list or other type, convert to tensor on the correct device
                stop = torch.tensor(stop, device=device, dtype=input_ids.dtype)
            
            # Per row: a bare True would end every sequence in the batch when row 0 finishes.
            if len(stop) <= input_ids.shape[1]:
                done = (input_ids[:, -len(stop):] == stop).all(dim=-1)
                return done if input_ids.shape[0] > 1 else bool(done[0])

        return False