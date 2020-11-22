# jubeatnet

_(Work In Progress)_

End to End Solution for parsing, modelling and visualizing jubeat songs and player fingering techniques.

## Modules

### core

The `jubeatnet.core.Pattern` class represents a pattern of notes in the jubeat 4x4 grid.

The `jubeatnet.core.JubeatChart` class represents a single jubeat chart in the game. In addition to containing metadata like title/artist/bpm, it also contains the pattern sequence data.

Each chart can be converted to a numpy array using `to_numpy_array`, including options for timecode unit/format and whether to include hold notes or not.

Each chart can be serialized to json using `to_json_string` and then deserialized using `from_json_string`.

### parser

The `parsers` module is able to read beatmap memos created by the community to create `JubeatChart` objects.

Each community wiki/site has a different way of annotating beatmaps. So there are different classes written for each site. Current sites are supported:

- [Cosmos Memo](https://w.atwiki.jp/cosmos_memo/) – `jubeatnet.parsers.CosmosMemoParser`
- Sonicy Memo (coming soon)

```python3
from jubeatnet.parsers import CosmosMemoParser
from pathlib import Path

memo = Path("./data/memo/heavenly_moon.txt")
parser = CosmosMemoParser()
chart = parser.parse(memo)
```

Parsers **are stateful**, so you should create a new parser object for each new beatmap you are parsing.