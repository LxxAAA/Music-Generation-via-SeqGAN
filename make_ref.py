#这个里面有和音乐相关的库 以及操作了
from music21 import *
import os
from copy import copy
import pickle

#读取数据
def load_data(file_path):
    ## load midi file using music21 library
    piece = converter.parse(file_path) #converter应该是music21里面的一个对象，用来转换， parse是解析的意思
    """
    # transpose all streams to C major. this process is to reduce the number of states
    # store the key of music before transposition.
    k = pre_piece.analyze('key')
    # save the interval of C and current key
    if k.mode == 'minor':
        i = interval.Interval(k.parallel.tonic, pitch.Pitch('C'))
    else:
        i = interval.Interval(k.tonic, pitch.Pitch('C'))
    # transpose the music using stored interval
    piece = pre_piece.transpose(i)
    # return transposed music
    """
    return piece


class preprocessing(object): #预处理类 我不知道这些是有什么用的， 是不是要找一个music21的教程
    def __init__(self):
        ## to collect and count unique chords, notes, octaves 和弦、音调、八度音阶
        # lists that store unique chords and octaves
        self.chords = [] #chord是和弦的意思
        self.chord_octaves = [] #和弦 八度
        
        #octaves是调子，从1到8， 是竖着的那个
        #chords 和 octaves 之间的区别在哪
        
        # lists for counting the number of times the chords and octaves appear
        # 用于计算和弦和八度音阶出现次数的列表， cnt应该是counting的意思把
        self.chords_cnt = [0] * len(self.chord_ref)
        self.chord_octaves_cnt = [0] * len(self.octave_ref)
        
        # the same thing about notes
        self.notes = []
        self.note_octaves = []
        self.notes_cnt = [0] * len(self.note_ref)
        self.note_octaves_cnt = [0] * len(self.note_octave_ref)

    def parsing(self, data_path): #parsing是解析的意思
        # load midi file
        piece = load_data(data_path)
        # all_parts is list of melody and chords, each is sequence of [start time, duration, octave, pitch, velocity]
        # 所有的部分 就是 旋律以及和弦的列表， 是（开始时间，持续时间，音调，音高，以及速度的）列表
        all_parts = []
        # for all parts in midi file (most of our data have two parts, melody and chords) 通常有两部分，旋律和和弦
        
        for part in piece.iter.activeElementList: #
            """ # check that the part is a piano song.
            # save the instrument name.
            try:
                track_name = part[0].bestName()
            except AttributeError:
                track_name = 'None'
            part_tuples.append(track_name)

            """
            # part_tuples is sequence of [start time, duration, octave, pitch] 是all part列表的元素吧
            part_tuples = []
            for event in part._elements:
                # if chords or notes exist recursive (sometimes this happened in a complicated piano song file)
                # 如果和弦 和 章节递归存在
                if event.isStream:
                    _part_tuples = []
                    for i in event._elements:
                        _part_tuples = self.streaming(event, i, _part_tuples)
                    all_parts.append(_part_tuples)#直接加入？
                # normal case
                # 平常的情况
                else:
                    part_tuples = self.streaming(part, event, part_tuples)
            if part_tuples != []:
                all_parts.append(part_tuples)
        return all_parts

    def streaming(self, part, event, part_tuples) #流， 给一个part 
        # find the set of chords and octaves
        # save start time
        for y in event.contextSites():
            if y[0] is part:
                offset = y[1]
        # if current event is chord 如果当前时间是 chord
        if getattr(event, 'isChord', None) and event.isChord:
            # chord pitch ordering 对和弦音高 排序
            octaves = []
            for pitch in event.pitches:
                octaves.append(pitch.octave)
            sort_idx = [i[0] for i in sorted(enumerate(event.pitchClasses), key=lambda x: x[1])] #这是排序吧
            octaves = [x for (y, x) in sorted(zip(sort_idx, octaves))] #音调 也排序 sort
            # if current chord or octave is unique until now, add it to the list 如果当前 和弦 音调是 唯一的， 之间没出现过， 加入 list
            if event.orderedPitchClasses not in self.chords:
                self.chords.append(event.orderedPitchClasses)
            if octaves not in self.chord_octaves:
                self.chord_octaves.append(octaves)

        # if current event is note 如果当前事件是一个 音符 （是指 单音么？）
        if getattr(event, 'isNote', None) and event.isNote:
            # find set of octaves and pitches of note 找到这个单音的 音调 和 音高， 并且加入self note
            if event.pitch.octave not in self.note_octaves:
                self.note_octaves.append(event.pitch.octave)
            if event.pitchClass not in self.notes:
                self.notes.append(event.pitchClass)

        # if current event is rest 如果当前事件 是空
        if getattr(event, 'isRest', None) and event.isRest:
            part_tuples.append([offset, event.quarterLength, 0, 0, 0])
        return part_tuples


if __name__ == "__main__":

    # print set of octaves and chords
    a = preprocessing()
    data_dir = './Nottingham/all/'
    dataset = []
    for file in os.listdir(data_dir):
        print(file)
        seq = a.parsing(data_dir + file)
        print('notes: ', a.notes)
        print('len(notes): ', len(a.notes))
        print('note_octaves: ', a.note_octaves)
        print('len(note_octaves): ', len(a.note_octaves))
        print('chords: ', a.chords)
        print('len(chords): ', len(a.chords))
        print('chord_octaves: ', a.chord_octaves)
        print('len(chord_octaves): ', len(a.chord_octaves))
        print('\n')
