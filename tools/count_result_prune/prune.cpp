#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>

void prune(char *filename, unsigned long bin)
{
	FILE *fin = fopen(filename, "r");

	if (NULL == fin) {
		fprintf(stderr, "Error: cannot open file %s.\n", filename);
		exit(1);
	}

	std::string name_str(filename);
	std::string output_name = name_str.substr(0, name_str.rfind(".")) + "_pruned" + ".txt";

	//printf("output_name: %s\n", output_name.c_str()); exit(1);//test

	FILE *fout = fopen(output_name.c_str(), "w");
	if (NULL == fout) {
		fprintf(stderr, "Error: cannot open file %s.\n", output_name.c_str());
		exit(1);
	}

	unsigned long line = 0;
	unsigned long id;
	unsigned long count;
	unsigned long sum = 0;

	while (fscanf(fin, "%lu%lu", &id, &count) != EOF) {
		++line;
		sum += count;
		if (0 == line % bin) {
			unsigned long avg = sum / bin;
			fprintf(fout, "%lu %lu\n", line / bin, avg);
			sum = 0;
		}
	}

	fclose(fin);
	fclose(fout);
}

int main(int argc, char *argv[])
{
	char *filename = nullptr;
	unsigned long bin;

	if (argc > 2) {
		filename = argv[1];
		bin = strtoul(argv[2], NULL, 0);
	} else {
		puts("Usage: /prune <data> <bin_width>");
		exit(1);
	}

	prune(filename, bin);

	return 0;
}
