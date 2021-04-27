#!/usr/bin/env python
# coding: utf-8



from keras.models import load_model
from collections import Counter
from keras.utils import to_categorical
import numpy as np
from music21 import converter, instrument, note, chord, stream

def get_data(filename):
    """从文件中获取音符

    :param filename: [文件名]
    :type filename: [str]
    :return: [返回音符]
    :rtype: [list]
    """
    with open(filename) as f:
       all_notes = f.readlines()
    return [ note[:len(note)-1]  for note in all_notes]

def predict_next(X_predict, model):
    """通过前100个音符，预测下一个音符

    :param X_predict: [前100个音符]
    :type X_predict: [list]
    :return: [下一个音符的id]
    :rtype: [int]
    """
    prediction = model.predict(X_predict)
    index = np.argmax(prediction)
    return index





def generate_notes(X_one_hot, id_to_note, X_train,model, duration):
    """随机从X_one_hot抽取一个数据（长为100），然后进行predict，最后生成音乐

    :return: [note数组（['D5', '2.6', 'F#5', 'D3', ……]）]
    :rtype: [list]
    """
    # 随机从X_one_hot选择一个数据进行predict
    randindex = np.random.randint(0, len(X_one_hot) - 1)
    predict_input = X_one_hot[randindex]
    # music_output里面是一个数组，如['D5', '2.6', 'F#5', 'D3', 'E5', '2.6', 'G5', 'F#5']
    music_output = [id_to_note[id] for id in X_train[randindex]]
    # 产生长度为1000的音符序列
    for note_index in range(int(float(duration)*100)):
        prediction_input = np.reshape(predict_input, (1,X_one_hot.shape[1],X_one_hot.shape[2]))
        # 预测下一个音符id
        predict_index = predict_next(prediction_input,model)
        # 将id转换成音符
        music_note = id_to_note[predict_index]
        music_output.append(music_note)
        # X_one_hot.shape[-1] = 308
        one_hot_note = np.zeros(X_one_hot.shape[-1])
        one_hot_note[predict_index] = 1
        one_hot_note = np.reshape(one_hot_note,(1,X_one_hot.shape[-1]))
        # 重新构建LSTM的输入
        predict_input = np.concatenate((predict_input[1:],one_hot_note))
    return music_output




def generate_music(result_data, instr, filename):
    """生成mid音乐，然后进行保存

    :param result_data: [音符列表]
    :type result_data: [list]
    :param filename: [文件名]
    :type filename: [str]
    """
    result_data = [str(data) for data in result_data]
    offset = 0
    output_notes = []
    # 生成 Note（音符）或 Chord（和弦）对象
    for data in result_data:
        if ('.' in data) or data.isdigit():
            notes_in_chord = data.split('.')
            notes = []
            if instr == 'Flute':        
                output_notes.append(instrument.Flute())
            elif instr == 'Piano':
                output_notes.append(instrument.Piano())
            elif instr == 'Bass':
                output_notes.append(instrument.Bass())
            elif instr == 'Guitar':
                output_notes.append(instrument.Guitar())    
            elif instr == 'Saxophone':
                output_notes.append(instrument.Saxophone())                   
            elif instr == 'Violin':
                output_notes.append(instrument.Violin()) 
                
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))             
                #new_note.storedInstrument = instrument.Flute()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)

        else:
            if instr == 'Flute':        
                output_notes.append(instrument.Flute())
            elif instr == 'Piano':
                output_notes.append(instrument.Piano())
            elif instr == 'Bass':
                output_notes.append(instrument.Bass())
            elif instr == 'Guitar':
                output_notes.append(instrument.Guitar())    
            elif instr == 'Saxophone':
                output_notes.append(instrument.Saxophone())                   
            elif instr == 'Violin':
                output_notes.append(instrument.Violin()) 
            new_note = note.Note(data)
            new_note.offset = offset
            #new_note.storedInstrument = instrument.Flute()
            output_notes.append(new_note)
        offset += 1
    # 创建音乐流（Stream）
    midi_stream = stream.Stream(output_notes)
    # 写入 MIDI 文件
    midi_stream.write('midi', fp=filename+'.mid')   

def generate(instrument, filename, duration):
  
    # 从保存的数据集中获得数据
    all_notes = get_data("data.txt")
    model = load_model("weights-804-0.01.hdf5")
    counter = Counter(all_notes)
    note_count = sorted(counter.items(),key=lambda x : -x[1])
    notes,_ = zip(*note_count)
    # note到id的映射
    note_to_id = {note:id for id,note in enumerate(notes)}
    # id到note的映射
    id_to_note = {id:note for id,note in enumerate(notes)}
    # 构建X_train，目的是随机从X_one_hot选择一个数据，然后进行predict 
    X_train = []
    sequence_batch = 100
    for i in range(len(all_notes)-sequence_batch):
        X_pre = all_notes[i:i+sequence_batch]
        X_train.append([note_to_id[note] for note in X_pre])
    X_one_hot = to_categorical(X_train)
    predict_notes = generate_notes(X_one_hot,id_to_note,X_train,model,duration)
    generate_music(predict_notes, instr=instrument, filename=filename)
    
    #predict_notes = generate_notes()
    #generate(instrument='Guitar', filename='here')







