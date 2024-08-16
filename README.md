# Auto-Note
Machine learning assisted note generation and search database provided a audio recoding of a lecture.

## Reducing file size
lecture.mp3 is now only 7.3 mb
```console
ffmpeg -i audio.mp3 -vn -map_metadata -1 -ac 1 -c:a libopus -b:a 12k -application voip audio.ogg
```