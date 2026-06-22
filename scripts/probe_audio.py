"""클립에 오디오 트랙이 있는지 실측 (오디오 임베딩 설계 검증용, 일회성)."""

import glob

import av

clips = sorted(glob.glob("연기영상/clips/*.mp4"))
print(f"총 클립 수: {len(clips)}")

checked = 0
with_audio = 0
for path in clips:
    try:
        container = av.open(path)
        astreams = [s for s in container.streams if s.type == "audio"]
        if astreams:
            with_audio += 1
            if checked < 5:
                s = astreams[0]
                cc = s.codec_context
                print(
                    f"  [오디오 O] {path.split('/')[-1]} "
                    f"codec={cc.name} rate={cc.sample_rate} ch={cc.channels} "
                    f"dur={float(container.duration) / av.time_base:.1f}s"
                )
        else:
            print(f"  [오디오 X] {path.split('/')[-1]}")
        container.close()
        checked += 1
    except Exception as e:
        print(f"  [에러] {path}: {e}")

print(f"\n오디오 있는 클립: {with_audio}/{len(clips)}")
