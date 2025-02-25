from gibberish.gibberish import GibberishDetector

def main():
    gb = GibberishDetector()
    gb.train(max_files=200)


if __name__ == "__main__":
    main()
