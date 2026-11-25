| Case | What it tests | Expected | Got | Result |
|---|---|---|---|---|
| `silence` | 10 s of digital silence | NoSpeechError | NoSpeechError | PASS |
| `music_only` | synthetic chord, no speech | NoSpeechError | NoSpeechError | PASS |
| `white_noise` | 10 s of noise | NoSpeechError | NoSpeechError | PASS |
| `short_3s` | 3-second excerpt of speech | ok | ok | PASS |
| `noisy_10dB` | speech with white noise at 10 dB SNR | ok | ok | PASS |
| `very_quiet` | speech at -26 dB | ok | ok | PASS |
| `fast_speech` | speech sped up 1.5x | ok | ok | PASS |
| `video_with_speech` | MP4 video + speech | ok | ok | PASS |
| `video_no_audio` | video with no audio track | InputError | InputError | PASS |
| `hindi_source` | Hindi speech given as the source | UnsupportedLanguageError | UnsupportedLanguageError | PASS |
| `corrupt_file` | random bytes named .mp3 | InputError | InputError | PASS |
| `empty_file` | zero-byte file | InputError | InputError | PASS |
| `too_long_31min` | 31 minutes (limit is 30) | InputError | InputError | PASS |
| `missing_file` | path that does not exist | InputError | InputError | PASS |
| `bad_language` | unsupported target language code 'xx' | InputError | InputError | PASS |

15 / 15 cases behaved as expected.
