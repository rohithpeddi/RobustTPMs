import wandb
import json
import csv
import os
import pandas as pd
from constants import *
from utils import mkdir_p

api = wandb.Api()

ROSPN = "ROSPN"
ROSPN_O = "ROSPN-O"

entities = ["rohithpeddi", "utd-ml-pgm"]
projects = [ROSPN, ROSPN_O]

mkdir_p("sup_results")
mkdir_p("results")

directory = 'sup_results'


def generate_csv():
	for i in range(2):
		entity = entities[i]
		project = projects[i]

		runs = api.runs(entity + "/" + project)
		for run in runs:
			summary_list = []
			summary_list.append(run.summary._json_dict)

			tables = list(summary_list[0].keys())
			tables.sort()

			for table in tables:
				print("-------------------------------------------------------------")
				print("Writing the information from project {}, file {}".format(project, table))

				for dataset_name in DEBD_DATASETS:
					if dataset_name in table:
						if "-LL" in table:
							raw_data_file = open("sup_results/{}_{}.csv".format(dataset_name, "LL"), "a")
						elif "-CLL" in table:
							for evidence_percentage in EVIDENCE_PERCENTAGES:
								if str(evidence_percentage) in table:
									raw_data_file = open(
										"sup_results/{}_{}_{}.csv".format(dataset_name, "CLL", evidence_percentage),
										"a")
						else:
							continue

						csv_writer = csv.writer(raw_data_file)

						if project == ROSPN:
							csv_writer.writerow(
								["MODEL", "C", "LS-1", "LS-3", "LS-5", "RLS-1", "RLS-3", "RLS-5",
								 "AV-1", "AV-3", "AV-5", "W-1", "W-3", "W-5"])

						meta = json.load(run.file(summary_list[0][table]['path']).download())
						table_data = meta["data"]

						for id in range(len(table_data)):
							data_row = table_data[id]
							if project == ROSPN_O and id == 0:
								print("Ignoring : {}".format(data_row))
								continue

							modified_data_row = []

							# Change attack name
							attack_type = data_row[0]
							if data_row[0] == LOCAL_SEARCH:
								if project == ROSPN:
									attack_type = "RALS"
								elif project == ROSPN_O:
									attack_type = "ALS"
								attack_type = "{}-{}".format(attack_type, data_row[1])
							elif data_row[0] == RESTRICTED_LOCAL_SEARCH:
								if project == ROSPN:
									attack_type = "RARLS"
								elif project == ROSPN_O:
									attack_type = "ARLS"
								attack_type = "{}-{}".format(attack_type, data_row[1])
							elif data_row[0] == CLEAN:
								attack_type = "CLEAN"
							data_row[0] = attack_type

							# Adding attack type and perturbations
							modified_data_row.append(attack_type)

							# Adding only mean columns
							for didx in range(2, len(data_row)):
								if didx % 2 == 0:
									modified_data_row.append(abs(round(data_row[didx], 1)))
							csv_writer.writerow(modified_data_row)


def generate_results_csv():
	for i in range(2):
		entity = entities[i]
		project = projects[i]

		runs = api.runs(entity + "/" + project)
		for run in runs:
			summary_list = []
			summary_list.append(run.summary._json_dict)

			tables = list(summary_list[0].keys())
			tables.sort()

			for table in tables:
				print("-------------------------------------------------------------")
				print("Writing the information from project {}, file {}".format(project, table))

				for dataset_name in DEBD_DATASETS:
					if dataset_name in table:
						if "-LL" in table:
							raw_data_file = open("results/{}_{}.csv".format(dataset_name, "LL"), "a")
						elif "-CLL" in table:
							for evidence_percentage in EVIDENCE_PERCENTAGES:
								if str(evidence_percentage) in table:
									raw_data_file = open(
										"results/{}_{}_{}.csv".format(dataset_name, "CLL", evidence_percentage), "a")
						else:
							continue

						csv_writer = csv.writer(raw_data_file)

						if project == ROSPN:
							csv_writer.writerow(
								["MODEL", "C", "LS-1", "LS-3", "LS-5", "RLS-1", "RLS-3", "RLS-5",
								 "AV-1", "AV-3", "AV-5", "W-1", "W-3", "W-5"])

						meta = json.load(run.file(summary_list[0][table]['path']).download())
						table_data = meta["data"]

						for id in range(len(table_data)):
							data_row = table_data[id]
							if project == ROSPN_O and id == 0:
								print("Ignoring : {}".format(data_row))
								continue

							modified_data_row = []

							# Change attack name
							attack_type = data_row[0]
							if data_row[0] == LOCAL_SEARCH:
								if project == ROSPN:
									attack_type = "RALS"
								elif project == ROSPN_O:
									attack_type = "ALS"
								attack_type = "{}-{}".format(attack_type, data_row[1])
							elif data_row[0] == RESTRICTED_LOCAL_SEARCH:
								if project == ROSPN:
									attack_type = "RARLS"
								elif project == ROSPN_O:
									attack_type = "ARLS"
								attack_type = "{}-{}".format(attack_type, data_row[1])
							elif data_row[0] == CLEAN:
								attack_type = "CLEAN"
							data_row[0] = attack_type

							# Adding attack type and perturbations
							modified_data_row.append(attack_type)

							# Adding only mean columns
							for didx in range(2, len(data_row)):
								if didx % 2 == 0:
									modified_data_row.append(abs(round(data_row[didx], 2)))
							csv_writer.writerow(modified_data_row)


def get_details(filename):
	name, description, type = None, None, None
	for dataset_name in DEBD_DATASETS:
		if dataset_name in filename:
			name = DEBD_display_name[dataset_name]
		if "_LL" in filename:
			description = "negative log likelihood scores"
			type = "LL"
		elif "_CLL" in filename:
			for evidence_percentage in EVIDENCE_PERCENTAGES:
				if str(evidence_percentage) in filename:
					description = "conditional log likelihood scores with evidence percentage {}".format(
						evidence_percentage * 100)
					type = "CLL-{}".format(evidence_percentage)
	return name, description, type


def generate_supplementary_latex_code(filename):
	directory = 'sup_results'
	file = os.path.join(directory, filename)
	print("Converting to file {} to latex".format(filename))
	name, description, type = get_details(filename)
	if os.path.isfile(file):
		df = pd.read_csv(file, header=None)
		with open("sup_table.tex", "a") as f:
			f.write("\n")
			f.write("\\begin{table}\n")
			f.write("\\centering \n")
			f.write("\\caption{\\label{tab:" + name + "-" + type + "}" + name + "-" + description + "}\n")
			f.write("\\begin{tabular}{ |" + " | ".join(["c"] * len(df.columns)) + " |}\n")
			f.write("\\hline \n")
			for i, row in df.iterrows():
				f.write(" & ".join([str(x) for x in row.values]) + " \\\\\n")
				if i == 0:
					f.write("\\hline \n")
			f.write("\\hline \n")
			f.write("\\end{tabular}\n")
			f.write("\\end{table}\n")


def generate_suplementary_latex():
	for type in ["LL", "CLL"]:
		for dataset_name in DEBD_DATASETS:
			if type == "LL":
				generate_supplementary_latex_code("{}_{}.csv".format(dataset_name, type))
			elif type == "CLL":
				for evidence_percentage in EVIDENCE_PERCENTAGES:
					generate_supplementary_latex_code("{}_{}_{}.csv".format(dataset_name, type, evidence_percentage))


def generate_results_code(filename):
	directory = 'results'
	mkdir_p("results_final")
	file = os.path.join(directory, filename)
	print("Parsing file {} to generate final csv".format(filename))
	name, description, type = get_details(filename)
	if os.path.isfile(file):
		df = pd.read_csv(file, header=[0])
		with open("results_final/{}.csv".format(type), "a") as f:
			result_row = [name]
			# D = 1
			# Clean SPN [CLEAN]
			clean_row = [df.iloc[0]['C'], df.iloc[0]['LS-1'], df.iloc[0]['RLS-1']]
			# Adv SPN [ALS]
			adv_row = [df.iloc[7]['C'], df.iloc[7]['LS-1'], df.iloc[7]['RLS-1']]
			# Regularized Adv SPN [RALS]
			radv_row = [df.iloc[1]['C'], df.iloc[1]['LS-1'], df.iloc[1]['RLS-1']]
			result_row.extend(clean_row)
			result_row.extend(adv_row)
			result_row.extend(radv_row)

			# D = 3
			# Clean SPN [CLEAN]
			clean_row = [df.iloc[0]['C'], df.iloc[0]['LS-3'], df.iloc[0]['RLS-3']]
			# Adv SPN [ALS]
			adv_row = [df.iloc[9]['C'], df.iloc[9]['LS-3'], df.iloc[9]['RLS-3']]
			# Regularized Adv SPN [RALS]
			radv_row = [df.iloc[3]['C'], df.iloc[3]['LS-3'], df.iloc[3]['RLS-3']]
			result_row.extend(clean_row)
			result_row.extend(adv_row)
			result_row.extend(radv_row)

			# # D = 5
			# # Clean SPN [CLEAN]
			# clean_row = [df.iloc[0]['C'], df.iloc[0]['LS-5'], df.iloc[0]['RLS-5']]
			# # Adv SPN [ALS]
			# adv_row = [df.iloc[11]['C'], df.iloc[11]['LS-5'], df.iloc[11]['RLS-5']]
			# # Regularized Adv SPN [RALS]
			# radv_row = [df.iloc[5]['C'], df.iloc[5]['LS-5'], df.iloc[5]['RLS-5']]
			# result_row.extend(clean_row)
			# result_row.extend(adv_row)
			# result_row.extend(radv_row)

			print("Storing model scores {}".format([df.iloc[0]['MODEL'], df.iloc[7]['MODEL'], df.iloc[1]['MODEL'],
													df.iloc[0]['MODEL'], df.iloc[9]['MODEL'], df.iloc[3]['MODEL'],
													df.iloc[0]['MODEL'], df.iloc[11]['MODEL'], df.iloc[5]['MODEL']]))
			csv_writer = csv.writer(f)
			csv_writer.writerow(result_row)


def generate_results():
	for type in ["LL", "CLL"]:
		for dataset_name in DEBD_DATASETS:
			if type == "LL":
				generate_results_code("{}_{}.csv".format(dataset_name, type))
			elif type == "CLL":
				for evidence_percentage in EVIDENCE_PERCENTAGES:
					generate_results_code("{}_{}_{}.csv".format(dataset_name, type, evidence_percentage))


def generate_results_latex_code(filename, type):
	directory = 'results_final'
	mkdir_p("results_final/tex/")
	file = os.path.join(directory, filename)
	print("Converting to file {} to latex".format(filename))
	if os.path.isfile(file):
		df = pd.read_csv(file, header=None)
		with open("results_final/tex/{}_table.tex".format(type), "a") as f:
			f.write("\n")
			f.write("\\begin{table*}[t]\n")
			f.write("\\begin{center} \n")
			f.write("\\begin{tabular}{ |" + " | ".join(["c"] * len(df.columns)) + " |}\n")
			f.write("\\hline \n")
			f.write(" & \\multicolumn{9}{|c|}{D=1} & \\multicolumn{9}{|c|}{D=3} \\\\\n")
			f.write("\\hline \n")
			f.write(
				" & \\multicolumn{3}{|c|}{SPN}  & \\multicolumn{3}{|c|}{SPN\'} & \\multicolumn{3}{|c|}{SPN\"}  & \\multicolumn{3}{|c|}{SPN}  & \\multicolumn{3}{|c|}{SPN\'} & \\multicolumn{3}{|c|}{SPN\"}  \\\\\n")
			f.write("\\hline \n")
			f.write(
				" \\multicolumn{1}{|c|}{dataset} & \\multicolumn{1}{|c|}{t} & \\multicolumn{1}{|c|}{t\'} & \\multicolumn{1}{|c|}{t\"} & \\multicolumn{1}{|c|}{t} & \\multicolumn{1}{|c|}{t\'} & \\multicolumn{1}{|c|}{t\"} & \\multicolumn{1}{|c|}{t} & \\multicolumn{1}{|c|}{t\'} & \\multicolumn{1}{|c|}{t\"} & \\multicolumn{1}{|c|}{t} & \\multicolumn{1}{|c|}{t\'} & \\multicolumn{1}{|c|}{t\"} & \\multicolumn{1}{|c|}{t} & \\multicolumn{1}{|c|}{t\'} & \\multicolumn{1}{|c|}{t\"} & \\multicolumn{1}{|c|}{t} & \\multicolumn{1}{|c|}{t\'} & \\multicolumn{1}{|c|}{t\"} \\\\\n")
			f.write("\\hline \n")
			# f.write("\\hline \n")
			# f.write(" & \\multicolumn{9}{|c|}{D=3} \\\\\n")
			# f.write("\\hline \n")
			# f.write(" & \\multicolumn{3}{|c|}{SPN}  & \\multicolumn{3}{|c|}{SPN\'} & \\multicolumn{3}{|c|}{SPN\"}  \\\\\n")
			# f.write("\\hline \n")
			# f.write(" \\multicolumn{1}{|c|}{dataset} & \\multicolumn{1}{|c|}{t} & \\multicolumn{1}{|c|}{t\'} & \\multicolumn{1}{|c|}{t\"} & \\multicolumn{1}{|c|}{t} & \\multicolumn{1}{|c|}{t\'} & \\multicolumn{1}{|c|}{t\"} & \\multicolumn{1}{|c|}{t} & \\multicolumn{1}{|c|}{t\'} & \\multicolumn{1}{|c|}{t\"} \\\\\n")
			# f.write("\\hline \n")
			for i, row in df.iterrows():
				f.write(" & ".join([str(x) for x in row.values]) + " \\\\\n")
			f.write("\\hline \n")
			f.write("\\end{tabular}\n")
			f.write("\\end{center} \n")
			f.write("\\end{table*}\n")


def generate_results_latex():
	for type in ["LL", "CLL"]:
		if type == "LL":
			generate_results_latex_code("{}.csv".format(type), type)
		elif type == "CLL":
			for evidence_percentage in EVIDENCE_PERCENTAGES:
				generate_results_latex_code("{}-{}.csv".format(type, evidence_percentage), type="{}-{}".format(type, evidence_percentage))


if __name__ == '__main__':
	# generate_csv()
	# generate_latex()
	generate_results_csv()
	generate_results()
	generate_results_latex()