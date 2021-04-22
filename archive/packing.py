import wave, struct, math

sampleRate = 44100.0 # hertz
duration = 1.0       # seconds
frequency = 440.0    # hertz

frequencies = [440, 3200, 5600, 7400, 10100, 12200, 15000, 16050, 19010, 21030]

wavef = wave.open('sound.wav','w')
wavef.setnchannels(1) # mono
wavef.setsampwidth(2) 
wavef.setframerate(sampleRate)

for i in range(int(duration * sampleRate)):
    value = 0
    for frequency in frequencies:
        value += int(32767.0*math.cos(frequency*math.pi*float(i)/float(sampleRate)))
    # value = int(32767.0*math.cos(frequency*math.pi*float(i)/float(sampleRate)))
    data = struct.pack('<l', value)
    wavef.writeframesraw( data )

# wavef.writeframes()
wavef.close()

