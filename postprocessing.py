#后期处理
from music21 import *
import pickle
from random import randint
from six.moves import xrange

path = '/music/seqGAN'

with open( path+'/dataset/chords', 'rb') as fp:
    chords_ref = pickle.load(fp)
with open( path+'/dataset/octaves', 'rb') as fp:
    octaves_ref = pickle.load(fp)

# map sequence of token (integer) to sequence of [melody duration, octave, key, velocity, chord duration, octave, key, velocity]
# 映射 列表 ——记号的列表（旋律持续时间， 八度， key ，速率，， 和弦持续， 音调， key， 速率）
def inverse_mapping(tokens):
    output = []
    # load reference list
    with open( path+'/dataset/tokens', 'rb') as fp:
        tokens_ref = pickle.load(fp)
    # inverse mapping : token is index of tokens_ref
    for token in tokens:
        output.append(tokens_ref[token])
    return output
# now the length of sequence is still 20, and length of each element is 8

# split melody and chords
def split(seq):
    melody = []
    chords = []
    for token in seq:
        # in every token, first four elements are melody and last four elements are chord
        # 对于每一个符号，前四个都是旋律， 而后四个都是和弦， 不明白。。。
        melody.append(token[0:3])
        chords.append(token[3:])
    return melody, chords

# make note from given token [duration, octave, key, velocity]
#对于给定的 符号（持续时间， 八度， key， 速率）来制作 单音
def make_event(token):
    # if token is rest
    # 如果是 空
    if token[2]==0:
        r = note.Rest()
        r.duration.quarterLength = token[0]
        event = r# midi.translate.noteToMidiEvents(r)
    # if token is note
    # 如果是 单音
    elif 0 < token[2] < 13:
        p = convert_pitch(token)
        n = note.Note(p[0])
        n.volume.velocity = 80
        n.duration.quarterLength = token[0]
        event = n#midi.translate.noteToMidiEvents(n)
    # if token is chord
    # 如果是 和弦
    else:
        p = convert_pitch(token)
        c = chord.Chord(p)
        c.volume.velocity = 80
        c.duration.quarterLength = token[0]
        event = c#midi.translate.chordToMidiEvents(c)
    return event

# convert (octave and key) to pitch for midi file
#转换为音高
def convert_pitch(token):
    # change elements of list from float to integer
    octave_ind = int(token[1])
    key_ind = int(token[2])
    # list of scale (C)
    scale = ['C#','D','D#','E','F','F#','G','G#','A','A#','B','C']
    # find octave and key
    octave = octaves_ref[octave_ind]
    key = chords_ref[key_ind]
    # check the number of key in chord is same in octave
    #assert len(octave) == len(key)
    # convert
    # convert octave and key to pitch string
    p = []
    for i in xrange(len(key)):
        p.append(scale[key[i]]+str(octave[i]))
    return p

def main(DATA, num_sample, epoch):
    # load sequence file
    #with open('./dataset/train', 'rb') as fp:
    with open(DATA, 'rb') as fp:
        seq = pickle.load(fp)

    for sample in xrange(num_sample):
        # select random sample of sequence
        seq_idx = randint(0,len(seq)-1)
        data = seq[seq_idx]
        # assumption : data is one sequence list, len(seq)=100, element of seq is integer

        sequence = inverse_mapping(data)
        melody, chords = split(sequence)

        all_parts = stream.Stream()

        # make melody stream
        part_melody = stream.Part()
        for token in melody:
            # skip dummy rest
            if token != [0, 0, 0]:
                event = make_event(token)
                # append event to part of melody
                part_melody.append(event)

        # make chord stream
        part_chord = stream.Part()
        chk_first = 1
        offset = 0
        for i in xrange(len(chords)):
            # skip dummy rest
            if chords[i] != [0, 0, 0]:
                # match fist start time of chord
                if chk_first == 1:
                    offset = part_melody[i].offset
                    chk_first = 0
                event = make_event(chords[i])
                # append event to part of chord
                part_chord.append(event)
                part_chord[-1].offset += offset


        all_parts.append([part_melody, part_chord])
    fp = all_parts.write('midi', path+'/midi/Ep_' + str(epoch) + '_test_' + str(sample) +'.mid')


if __name__ == "__main__":
    main(path+'/dataset/train',1,1)
