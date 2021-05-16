# Text to speech toolkits
import speech_recognition as sr
import pyttsx3
from pydub import AudioSegment
from pydub.utils import make_chunks
import wave
import pyaudio
# Generator Natural Langugae Tool Kit
import nltk
from nltk.data import load
from nltk import CFG
from nltk.grammar import is_nonterminal
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.corpus import stopwords
# Other Toolkits
import random
import sys  # print logging save to file - pip install os-sys
import re
from Arduino import Arduino
import time

# Global TTS variables
listener = sr.Recognizer()
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
poetry_voice_rate = 120
normal_voice_rate = 160

# Arduino Initialise
red_led_pin = 5                             # recording
white_led_pin = 6                           # stored
choice_servo_pin = 3                        # source text choice  - 40 recording, 80 bible, 130 poe, 180 sh
type_servo_pin = 4                          # poetry types          - 20 haiku, 70 short, 110 medium, 160 long
button_pin = 2                              # stop recording
baud = '115200'
port = '/dev/ttyACM0'                      #  '/dev/ttyACM0' - on RPI or 'COM3' - on windows
board = Arduino(baud, port=port)

board.pinMode(red_led_pin, "OUTPUT")        # REC
board.pinMode(white_led_pin, "OUTPUT")      # Store
board.Servos.attach(choice_servo_pin)       # servo 1
board.Servos.attach(type_servo_pin)         # servo 2
stop_button = board.analogRead(button_pin)  # stop button analog 0 or 1000


# ----------------------------------------------- Generator Setup ----------------------------------------------
class Text:
    def __init__(self, raw_text):
        self.text_array = nltk.word_tokenize(raw_text)
        self.POS_buckets = {}
        tagged_text_array = nltk.pos_tag(self.text_array)
        self.tags = load('help/tagsets/upenn_tagset.pickle')
        for tag in self.tags:
            self.POS_buckets[tag] = []
        for tuple in tagged_text_array:
            self.POS_buckets[tuple[1]].append(tuple[0].lower())
        self.before = {}
        self.after = {}
        for word in self.text_array:
            self.before[word] = []
            self.after[word] = []
        for i in range(len(self.text_array)):  # range goes to one less than given value
            if i > 0:
                self.before[self.text_array[i]].append(self.text_array[i - 1])
            if i < len(self.text_array) - 1:
                self.after[self.text_array[i]].append(self.text_array[i + 1])

    # return list of two word collocation lists
    def get_collocations(self):
        ignored_words = stopwords.words('english')
        finder = BigramCollocationFinder.from_words(self.text_array, 2)
        finder.apply_freq_filter(3)
        finder.apply_word_filter(lambda w: len(w) < 3 or w.lower() in ignored_words)
        bigram_measures = BigramAssocMeasures()
        return finder.nbest(bigram_measures.likelihood_ratio, 40)


class Grammar:

    def __init__(self, haiku):

        # comment about what each part of speach is:
        """ CC   - conjunction: or, but, and, either
            CD   - number: one, two, three
            DT   - determiner: a, an, the, both, all, these, any, some
            EX   - the word 'there'
            IN   - preposition: in, of, with, for, under, among, upon, at
            JJ   - adjective: certain, curious, little, golden, other, offended
            JJS  - adjective: -est : best, loveliest, largest
            JJR  - adjective: -er : lerger, smaller, worse
            MD   - can, dare, should, will*, might, could, must
            NN   - common singular noun
            NNS  - common plural noun
            NNP  - proper singular noun
            NNPS - proper plural noun
            PDT  - all, both, quite, many, half
            PRP  - hers, her, himself, thy, us, it, I, him, you, they
            PRPP - possesive: his, mine, our, my, her, its, your
            RB   - adverb: very, not, here, there, first, just, down, again, beautifully, -ly
            RBR  - more
            RBS  - adverb superlative: -est
            RP   - participle: up, down, out, away, over, off
            TO   - the word 'to'
            UH   - interjection
            VB   - vocative verb: to ___
            VBD  - past verb: -ed : was*(freq. occur), had, dipped, were, said, seemed
            VBG  - present verb: -ing: trembling, trying, getting, running, swimming
            VBN  - past verb descriptive: crowded, mutated, fallen, lit, lost, forgtten
            VBP  - present verb: not -s: am, wish, make, know, do, find
            VBZ  - present verb: -s : is*, has, seems
            WDT  - what, which, that*
            WP   - who, what
            WRB  - how, whenever, where, why, when
        """

        # create base of cfg
        if not haiku:
            g = CFG.fromstring("""
            S -> NPS VPS | NPS VPS | NPS VPS | NPP VPP | VPO | NPO
            S -> NPS VPS | NPP VPP | NPS VPS

            NPS -> 'DT' 'NN' | 'DT' 'NN' | 'DT' 'JJ' 'NN' | 'DT' 'JJ' 'NN'
            NPS -> 'EX' 'the' 'NN' | 'the' 'JJS' 'NN'
            NPS -> 'she' | 'he' | 'it' | 'I'
            NPS -> NPS INP | INP NPS

            NPP -> 'the' 'NNS' | 'the' 'NNS' | 'NNS'
            NPP -> 'the' 'JJ' 'NNS'
            NPP -> 'they' | 'you' | 'we'

            VING -> 'VBG' | 'VBG' | 'RB' 'VBG'
            VBB -> 'VB' | 'VB' | 'VBP'

            SM -> 'is' | 'was' | 'has been'

            VPS -> SM 'VBN' | SM 'VBN' 'like the' 'JJ' 'NN'
            VPS -> SM VING | SM VING INP
            VPS -> SM VING 'like' 'DT' 'JJ' 'NN'
            VPS -> SM VING 'like a' 'NN' INP
            VPS -> SM 'as' 'JJ' 'as' 'JJ'
            VPS -> SM 'a' 'JJ' 'NN'
            VPS -> SM 'a' 'NN' INP
            VPS -> 'MD' 'have been' VING
            VPS -> 'is' 'JJ' 'and' 'JJ'
            VPS -> 'VBD' INP | 'RB' 'VBD'
            VPS -> SM 'VBD' 'like' 'DT' 'JJ' 'NN'
            VPS -> SM 'as' 'JJ' 'as the' 'NN'
            VPS -> 'VBD' 'NN' | 'VBD' 'DT' 'NN'
            VPS -> 'VBD' 'and' 'VBD' INP 'until' 'VBN'
            VPS -> VPS 'and' S
            VPS -> 'VBD' 'JJR' 'than' 'a' 'NN'
            VPS -> 'VBD' 'EX'
            VPS -> SM 'JJ' | SM 'VB' INP

            NPO -> 'a' 'NN' 'IN' 'NNP'
            NPO -> 'the' 'NN' 'IN' 'the' 'JJ' 'NNP'
            NPO -> 'the' 'NNS' 'IN' 'the' 'NN'

            VPO -> 'VBG' 'like' 'NNP' 'RP' 'DT' 'JJ' 'NN' 'IN' 'DT' 'NN'
            VPO -> 'has been' 'VBG' 'RP' 'and' 'VBG'

            PM -> 'are' | 'were' | 'have been'

            VPP -> PM VING | PM VING INP
            VPP -> PM VING 'like the' 'NNS' INP
            VPP -> PM 'as' 'JJ' 'as' NPS INP | PM 'JJ' 'like' 'NNS' | PM 'JJ' 'like' VBG 'NNS'
            VPP -> PM 'VBN' | PM 'VBN' INP
            VPP -> PM 'as' 'JJ' 'as' 'JJ' | PM 'as' 'JJ' 'as' 'VBG' 'NNS'
            VPP -> PM 'NNS' INP
            VPP -> PM 'JJ' 'NNS'
            VPP -> 'are' 'JJ' 'and' 'JJ'
            VPP -> 'VBD' INP | 'VBD' 'RP' INP
            VPP -> PM 'JJ' | PM 'VB' INP

            INP -> 'IN' 'DT' 'NN' | 'IN' 'the' 'NNS' | 'IN' 'the' 'JJ' 'NNS'
            INP -> 'IN' 'DT' 'NN' 'IN' 'DT' 'NN'
            INP -> 'IN' 'DT' 'JJ' 'NN' | 'RP' 'IN' 'DT' 'JJ' 'NN'
            INP -> 'RP' 'IN' 'DT' 'NN' | 'IN' 'JJ' 'NNS'
            INP -> 'IN' 'DT' 'NN' | 'RP' 'DT' 'NNS'
            """)

            # save grammar to self.cfg
            self.cfg = CFG.fromstring(str(g).split('\n')[1:])
            self.cfg._start = g.start()

        elif haiku:
            g2 = CFG.fromstring("""
                        S ->  'DT' 'JJ' 'NNS'
                        S -> 'VBD' 'NNS'
                        S ->  'NNS' 'VBD'

                        """)

            self.cfg = CFG.fromstring(str(g2).split('\n')[1:])
            self.cfg._start = g2.start()

    def gen_frame_line(self, nt):
        sentence = ''
        prods = random.sample(self.cfg.productions(lhs=nt), len(self.cfg.productions(lhs=nt)))
        valid = True
        for prod in prods:
            for sym in prod.rhs():
                if is_nonterminal(sym):
                    if len(self.cfg.productions(lhs=sym)) < 1:
                        valid = False
            if valid == True:
                for sym in prod.rhs():
                    if is_nonterminal(sym):
                        sentence += self.gen_frame_line(sym)
                    else:
                        sentence += sym + ' '
                break
        if valid == False:
            return " "  # ERROR
        else:
            return sentence  # removed capitalize


class Spot:

    def __init__(self, wop, line, column, content):
        if content == 'POS':
            self.word = ''
            self.POS = wop
            self.line = line
            self.column = column
            self.filled = False
            self.preset = False
        elif content == 'word':
            self.word = wop
            self.POS = ''
            self.line = line
            self.column = column
            self.filled = True
            self.preset = True
        else:
            print(" ")  # spot content error

    def fill(self, word):
        self.word = word
        self.filled = True

    def add_POS(self, pos):
        self.POS = pos


class Frame:

    def __init__(self, grammar, tags, length, haiku):
        self.lines = []
        repeat_line_array = nltk.word_tokenize(grammar.gen_frame_line(grammar.cfg.start()))
        if haiku == True:
            x = 3
            y = 2
        elif haiku == False:
            x = random.randint(0, length)
            y = random.randint(0, length)
        for i in range(length):
            if (i == x or i == y):
                spot_array = []
                j = 0
                noun_set = set(['he', 'she', 'it', 'I'])
                for wop in repeat_line_array:
                    if wop in set(tags):
                        spot = Spot(wop, i, j, 'POS')
                        if (wop in noun_set):
                            spot.add_POS('NN')
                        spot_array.append(spot)
                    else:
                        spot = Spot(wop, i, j, 'word')
                        spot_array.append(spot)
                    j += 1
                self.lines.append(spot_array)
            else:
                line_array = nltk.word_tokenize(grammar.gen_frame_line(grammar.cfg.start()))
                spot_array = []
                j = 0
                for wop in line_array:
                    if wop in set(tags):
                        spot = Spot(wop, i, j, 'POS')
                        spot_array.append(spot)
                    else:
                        spot = Spot(wop, i, j, 'word')
                        spot_array.append(spot)
                    j += 1
                self.lines.append(spot_array)

    def add_collocations(self, text):
        collocations = text.get_collocations()
        tagged_collocation_list = []
        for collocation in collocations:
            tagged_collocation_list.append(nltk.pos_tag(collocation))
        for tagged_collocation in tagged_collocation_list:
            POS_pair = [tagged_collocation[0][1], tagged_collocation[1][1]]
            word_pair = [tagged_collocation[0][0], tagged_collocation[1][0]]
        j = 0
        collocation_used = False
        for line in self.lines:
            if collocation_used == False:
                for i in range(len(line) - 1):  # 0 to line.length-2
                    if POS_pair == [line[i], line[i + 1]]:
                        line[i].fill(word_pair[0])
                        line[i + 1].fill(word_pair[1])
                        collocation_used = True
                        break
                j += 1

    def add_big_words(self, text):
        fdist = nltk.FreqDist(text.text_array)
        big_words = []
        for w in set(text.text_array):
            if len(w) > 6 and fdist[w] > 2:
                big_words.append(w)
        big_words_with_tags = nltk.pos_tag(big_words)
        big_word_buckets = {}
        for tag in text.tags:  # initialize list of words for each tag
            big_word_buckets[tag] = []
        for big_word_tuple in big_words_with_tags:
            big_word_buckets[big_word_tuple[1]].append(big_word_tuple[0])
        used_words = []
        for line in self.lines:
            for spot in line:
                if spot.filled == False and len(big_word_buckets[spot.POS]) > 0:
                    n = random.randint(0, len(big_word_buckets[spot.POS]) - 1)
                    big_word = big_word_buckets[spot.POS][n]
                    if big_word in set(used_words):
                        pass
                    else:
                        spot.fill(big_word)
                        used_words.append(big_word)

    def repeat_nouns(self, length):
        noun = ''
        for spot in self.lines[0]:
            if spot.POS == 'NN' and spot.filled == True:
                noun = spot.word
                break
        if noun == '': return
        for i in range(1, length):
            for spot in self.lines[i]:
                if spot.POS == 'NN' and spot.filled == False:
                    spot.fill(noun)
                    break

    def add_context_words(self, text):
        for line in self.lines:
            for spot in line:
                if spot.filled == True:
                    if spot.column > 0 and line[spot.column - 1].filled == False and spot.preset == False:
                        for before_word in text.before[spot.word]:
                            if line[spot.column - 1].POS == nltk.pos_tag([before_word])[0][1]:
                                line[spot.column - 1].fill(before_word)
                                break
                    if spot.column < len(line) - 1 and line[spot.column + 1].filled == False and spot.preset == False:
                        for after_word in text.after[spot.word]:
                            if line[spot.column + 1].POS == nltk.pos_tag([after_word])[0][1]:
                                line[spot.column + 1].fill(after_word)
                                break

    def add_random(self, text):
        while True:
            x = random.randint(0, 8)
            y = random.randint(0, len(self.lines[x]))
            spot = self.lines[x][y]
            if not spot.filled:
                n = random.randint(0, len(text.POS_buckets[spot.POS]) - 1)
                word = text.POS_buckets[spot.POS][n]
                spot.fill(word)
                return

    def add_first_unfilled(self, text):
        for line in self.lines:
            for spot in line:
                if spot.filled == False:
                    n = random.randint(0, len(text.POS_buckets[spot.POS]) - 1)
                    word = text.POS_buckets[spot.POS][n]
                    spot.fill(word)
                    break

    def fill_remaining(self, text):
        for line in self.lines:
            for spot in line:
                if not spot.filled:
                    n = random.randint(0, len(text.POS_buckets[spot.POS]) - 1)
                    word = text.POS_buckets[spot.POS][n]
                    spot.fill(word)

    def print(self):
        for line in self.lines:
            for spot in line:
                if spot.filled:
                    print(spot.word, end=" ")
                else:
                    print(spot.POS, end=" ")
            print()
        print()


# ------------------------------------------- TTS Setup -----------------------------------------------------------

class TTS:

    def take_command() -> object:
        try:
            with sr.Microphone() as source:
                command = 'No voice identified!\n'
                print('Gathering audio input!\n')
                voice = listener.listen(source) #, phrase_time_limit=10000)  # argument lo litsen a given time

                command = listener.recognize_google(voice)
                command = command.lower()

                if 'edgar' in command:
                    command = command.replace('edgar', '')
                    print(command + '\n')
        except:
            pass

        return command

    def talk(text):

        engine.say(text)
        engine.runAndWait()

    def record_audio():
        chunk = 1024  # Record in chunks of 1024 samples
        sample_format = pyaudio.paInt16  # 16 bits per sample
        channels = 2
        fs = 44100  # Record at 44100 samples per second
        seconds = 3600  # 1h
        filename = "recording_audio_temp.wav"

        p = pyaudio.PyAudio()  # Create an interface to PortAudio

        print('Recording')
        board.digitalWrite(red_led_pin, 'HIGH')
        stream = p.open(format=sample_format,
                        channels=channels,
                        rate=fs,
                        frames_per_buffer=chunk,
                        input=True)

        frames = []  # Initialize array to store frames

        # Store data in chunks for 3 seconds
        for i in range(0, int(fs / chunk * seconds)):
            data = stream.read(chunk)
            frames.append(data)
            board.analogRead(button_pin)
            if board.analogRead(button_pin) > 500:
                break

        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        # Terminate the PortAudio interface
        p.terminate()

        board.digitalWrite(red_led_pin, 'LOW')
        print('Finished recording')

        # Save the recorded data as a WAV file
        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))
        wf.close()

        print('Recording saved to file\n')

    def audio_file_chunks():
        source_audio = AudioSegment.from_file("recording_audio_temp.wav", "wav")
        chunk_length_ms = 10000
        chunks = make_chunks(source_audio, chunk_length_ms)
        for i, chunk in enumerate(chunks):
            chunk_name = "chunk{0}.wav".format(i)
            print("exporting", chunk_name)
            chunk.export(chunk_name, format="wav")
        i = 0
        with open('recording_audio_temp.txt', 'w+') as recording_temp:
            sys.stdout = recording_temp
            for chunk in chunks:
                chunk_silent = AudioSegment.silent(duration=10)
                audio_chunk = chunk_silent + chunk + chunk_silent
                audio_chunk.export("./chunk{0}.wav".format(i), bitrate='192k', format="wav")
                filename = 'chunk' + str(i) + '.wav'
                file = filename
                r = sr.Recognizer()
                with sr.AudioFile(file) as source:
                    audio = r.listen(source)
                try:
                    print(r.recognize_google(audio))
                except sr.UnknownValueError:
                    print(" ")
                except sr.RequestError as e:
                    print(" ".format(e))
                i += 1
        recording_temp.close()
        print('Exporting done')

    def read_last_poem():

        engine.setProperty('rate', poetry_voice_rate)  # slow down voice for poetry reading
        with open('poems_last.txt') as poems_last:
            string_without_line_breaks = ' '
            for line in poems_last:
                stripped_line = line.rstrip()
                string_without_line_breaks += stripped_line + ' '
        poem = str(string_without_line_breaks)  # read all lines
        poem_str = str(poem)  # read from start
        TTS.talk(poem_str)
        engine.setProperty('rate', normal_voice_rate)  # back to normal voice rate



# ----------------------------------------------------------- Functions Declaration -----------------------------------------------------------


class Functions:

    def clean_source_text(source):
        file = open(source, 'r+', encoding='utf-8')
        text = file.read()
        text = re.sub('[0-9]+', '', str(text))
        text = re.sub('\n ', '', str(text))
        text = re.sub('\n', ' ', str(text))
        text = re.sub('_', '', str(text))
        text = re.sub(':', '', str(text))
        text = re.sub("'s", '', str(text))
        text = re.sub("-", ' ', str(text))
        text = re.sub("— ", '', str(text))
        text = re.sub('\"', '', str(text))
        text = re.split('[.?!]', text)
        clean_sent = []
        for sent in text:
            clean_sent.append(sent)
        separator = ' '
        clean_text = separator.join(clean_sent)
        return clean_text

    def run_generator(source, length, haiku):
        file = open(source, 'r+', encoding='utf-8')
        read_source = file.read()
        text = Text(read_source)  # seperates words into POS buckets
        grammar = Grammar(haiku)  # makes CFG
        frame = Frame(grammar, text.tags, length, haiku)  # create "frame" of poem: list of lists of POS tags
        frame.add_collocations(text)
        frame.add_big_words(text)
        frame.repeat_nouns(length)
        for x in range(3):
            frame.add_context_words(text)
        frame.add_first_unfilled(text)
        frame.repeat_nouns(length)
        frame.add_context_words(text)
        frame.fill_remaining(text)
        with open('poems_last.txt', 'w+') as file_last_poem:
            sys.stdout = file_last_poem
            frame.print()
        file_last_poem.close()


    def save_poem_database():
        board.digitalWrite(white_led_pin, "HIGH")
        time.sleep(2)
        poem_database = open('poems_all.txt', 'a+')
        with open('poems_last.txt', 'r+') as file_last_poem:
            last_poem_temp = file_last_poem.readlines()
            poem_database.write('**#**\n')
            for line in last_poem_temp:
                poem_database.write(line)
        file_last_poem.close()
        poem_database.write('\r\n')
        poem_database.close()
        print('Poem stored!\n')
        time.sleep(2)
        board.digitalWrite(white_led_pin, "LOW")


    def retrieve_from_database():
        with open('poems_all.txt', "r") as fh:
            raw_text = fh.read()
            raw_text2 = raw_text.replace('\n', '')
            text_split = raw_text2.split('**#**')
            text_split.remove('')
            poem = random.choice(tuple(text_split))
            print(poem)
            engine.setProperty('rate', poetry_voice_rate)  # skip empty character item
            TTS.talk(poem)
            engine.setProperty('rate', normal_voice_rate)


    def run_edgar():
        command = TTS.take_command()  # take audio input
        if command != ('No voice identified!\n'):

            time.sleep(1)
            board.Servos.write(choice_servo_pin, 90)  # boot sequence
            board.Servos.write(type_servo_pin, 90)  # boot sequence
            time.sleep(0.5)
            board.Servos.write(type_servo_pin, 0)
            board.Servos.write(choice_servo_pin, 180)
            time.sleep(0.5)
            board.Servos.write(type_servo_pin, 180)
            board.Servos.write(choice_servo_pin, 0)
            time.sleep(0.5)
            board.Servos.write(type_servo_pin, 90)  # boot sequence
            board.Servos.write(choice_servo_pin, 90)  # boot sequence
            time.sleep(1)

            if 'listen' in command:
                TTS.record_audio()
                TTS.audio_file_chunks()

            elif 'read' in command:
                TTS.read_last_poem()

            elif 'save' in command:
                Functions.save_poem_database()

            elif 'retrieve' in command:
                Functions.retrieve_from_database()

            elif 'generate' in command:
                time.sleep(1)
                if 'shakespeare' in command:
                    source = 'shakespeare.txt'
                    print('Text Source: Shakespeare\n')
                    board.Servos.write(choice_servo_pin, 160)
                    time.sleep(1)
                elif 'bible' in command:
                    source = 'bible.txt'
                    print('Text Source: The Bible\n')
                    board.Servos.write(choice_servo_pin, 70)
                    time.sleep(1)
                elif 'recorded' in command:
                    source = 'recording_audio_temp.txt'
                    print('Text Source: Audio Recording\n')
                    board.Servos.write(choice_servo_pin, 20)
                    time.sleep(1)
                else:
                    source = 'poe_all.txt'
                    print('Text Source: Edgar AlLan Poe\n')
                    board.Servos.write(choice_servo_pin, 110)
                    time.sleep(1)

                if 'haiku' in command:
                    haiku = True
                    length = 3
                    print('Poetry type: haiku\n')
                    board.Servos.write(type_servo_pin, 20)
                    time.sleep(1)

                elif 'short' in command:
                    haiku = False
                    length = 4
                    print('Poetry type: Free Form\n')
                    print('Poetry length: Short\n')
                    board.Servos.write(type_servo_pin, 70)
                    time.sleep(1)

                elif 'long' in command:
                    haiku = False
                    length = 12
                    print('Poetry type: Free Form\n')
                    print('Poetry length: Long\n')
                    board.Servos.write(type_servo_pin, 160)
                    time.sleep(1)
                else:
                    haiku = False
                    length = 8
                    print('Poetry type: Free Form\n')
                    print('Poetry length: Medium\n')
                    board.Servos.write(type_servo_pin, 110)
                    time.sleep(1)

                Functions.run_generator(source, length, haiku)
                TTS.read_last_poem()
            else:
                TTS.talk('I was not able to understand the command.\n')
                print('I was not able to understand the command.\n')

            time.sleep(1)
            board.Servos.write(choice_servo_pin, 90)
            time.sleep(1)
            board.Servos.write(type_servo_pin, 90)
            time.sleep(1)



# ------------------------------------- Generator Main run -----------------------------------------------------------

if __name__ == "__main__":
    while True:
        Functions.run_edgar()