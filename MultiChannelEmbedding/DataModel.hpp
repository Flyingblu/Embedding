#pragma once
#include "Import.hpp"
#include "ModelConfig.hpp"

class DataModel
{
public:
	set<pair<pair<int, int>, int> >		check_data_train;
	set<pair<pair<int, int>, int> >		check_data_all;

public:
	vector<pair<pair<int, int>, int> >	data_train;
	vector<pair<pair<int, int>, int> >	data_dev_true;
	vector<pair<pair<int, int>, int> >	data_dev_false;
	vector<pair<pair<int, int>, int> >	data_test_true;
	vector<pair<pair<int, int>, int> >	data_test_false;

public:
	set<unsigned int>			set_entity;
	set<unsigned int>			set_relation;

public:
	vector<set<int>>	set_relation_tail;
	vector<set<int>>	set_relation_head;

public:
	vector<int>	relation_type;

public:
	vector<string>		entity_id_to_name;
	vector<string>		relation_id_to_name;
	map<string, int>	entity_name_to_id;
	map<string, int>	relation_name_to_id;

public:
	vector<double>		prob_head;
	vector<double>		prob_tail;
	vector<double>		relation_tph;
	vector<double>		relation_hpt;
	map<string, int>	count_entity;

public:
	map<int, map<int, int> >	tails;
	map<int, map<int, int> >	heads;

public:
	map<int, map<int, vector<int> > >     rel_heads;
	map<int, map<int, vector<int> > >     rel_tails;
	map<pair<int, int>, int>		     rel_finder;
	
public:
	int zeroshot_pointer;

public:
	DataModel(const Dataset& dataset)
	{
		load_training(dataset.base_dir + dataset.training);
		load_testing(dataset.base_dir + dataset.testing, data_test_true, data_test_false, dataset.self_false_sampling);
		relation_hpt.resize(set_relation.size());
		relation_tph.resize(set_relation.size());
		for(auto i=0; i!=set_relation.size(); ++i)
		{
			double sum = 0;
			double total = 0;
			for(auto ds=rel_heads[i].begin(); ds!=rel_heads[i].end(); ++ds)
			{
				++ sum;
				total += ds->second.size();
			}
			relation_tph[i] = total / sum;
		}
		for(auto i=0; i!=set_relation.size(); ++i)
		{
			double sum = 0;
			double total = 0;
			for(auto ds=rel_tails[i].begin(); ds!=rel_tails[i].end(); ++ds)
			{
				++ sum;
				total += ds->second.size();
			}
			relation_hpt[i] = total / sum;
		}

		zeroshot_pointer = set_entity.size();

		set_relation_head.resize(set_entity.size());
		set_relation_tail.resize(set_relation.size());
		prob_head.resize(set_entity.size());
		prob_tail.resize(set_entity.size());
		for(auto i=data_train.begin(); i!=data_train.end(); ++i)
		{
			++ prob_head[i->first.first];
			++ prob_tail[i->first.second];

			++ tails[i->second][i->first.first];
			++ heads[i->second][i->first.second];

			set_relation_head[i->second].insert(i->first.first);
			set_relation_tail[i->second].insert(i->first.second);
		}

#pragma omp parallel
#pragma ivdep
		for (auto elem = prob_head.begin(); elem != prob_head.end(); ++elem)
		{
			*elem /= data_train.size();
		}

#pragma omp parallel
#pragma ivdep
		for (auto elem = prob_tail.begin(); elem != prob_tail.end(); ++elem)
		{
			*elem /= data_train.size();
		}

		double threshold = 1.5;
		relation_type.resize(set_relation.size());

 		for(auto i=0; i<set_relation.size(); ++i)
		{
			if (relation_tph[i]<threshold && relation_hpt[i]<threshold)
			{
				relation_type[i] = 1;
			}
			else if (relation_hpt[i] <threshold && relation_tph[i] >= threshold)
			{
				relation_type[i] = 2;
			}
			else if (relation_hpt[i] >=threshold && relation_tph[i] < threshold)
			{
				relation_type[i] = 3;
			}
			else
			{
				relation_type[i] = 4;
			}
		}
	}

	DataModel(const Dataset& dataset, const string& file_zero_shot)
	{
		load_training(dataset.base_dir + dataset.training);

		relation_hpt.resize(set_relation.size());
		relation_tph.resize(set_relation.size());
		for (auto i = 0; i != set_relation.size(); ++i)
		{
			double sum = 0;
			double total = 0;
			for (auto ds = rel_heads[i].begin(); ds != rel_heads[i].end(); ++ds)
			{
				++sum;
				total += ds->second.size();
			}
			relation_tph[i] = total / sum;
		}
		for (auto i = 0; i != set_relation.size(); ++i)
		{
			double sum = 0;
			double total = 0;
			for (auto ds = rel_tails[i].begin(); ds != rel_tails[i].end(); ++ds)
			{
				++sum;
				total += ds->second.size();
			}
			relation_hpt[i] = total / sum;
		}

		zeroshot_pointer = set_entity.size();
		load_testing(dataset.base_dir + dataset.developing, data_dev_true, data_dev_false, dataset.self_false_sampling);
		load_testing(dataset.base_dir + dataset.testing, data_dev_true, data_dev_false, dataset.self_false_sampling);
		load_testing(file_zero_shot, data_test_true, data_test_false, dataset.self_false_sampling);

		set_relation_head.resize(set_entity.size());
		set_relation_tail.resize(set_relation.size());
		prob_head.resize(set_entity.size());
		prob_tail.resize(set_entity.size());
		for (auto i = data_train.begin(); i != data_train.end(); ++i)
		{
			++prob_head[i->first.first];
			++prob_tail[i->first.second];

			++tails[i->second][i->first.first];
			++heads[i->second][i->first.second];

			set_relation_head[i->second].insert(i->first.first);
			set_relation_tail[i->second].insert(i->first.second);
		}

		for (auto & elem : prob_head)
		{
			elem /= data_train.size();
		}

		for (auto & elem : prob_tail)
		{
			elem /= data_train.size();
		}

		double threshold = 1.5;
		relation_type.resize(set_relation.size());
		for (auto i = 0; i < set_relation.size(); ++i)
		{
			if (relation_tph[i] < threshold && relation_hpt[i] < threshold)
			{
				relation_type[i] = 1;
			}
			else if (relation_hpt[i] < threshold && relation_tph[i] >= threshold)
			{
				relation_type[i] = 2;
			}
			else if (relation_hpt[i] >= threshold && relation_tph[i] < threshold)
			{
				relation_type[i] = 3;
			}
			else
			{
				relation_type[i] = 4;
			}
		}
	}

	void load_training(const string& filename)
	{
		ifstream triple_file(filename, ios_base::binary);

		size_t triple_size;
		triple_file.read((char*)&triple_size, sizeof(size_t));
		data_train.reserve(triple_size);
		for (unsigned int i = 0; i < triple_size && triple_file; ++i)
		{
			unsigned int tri_arr[3];
			triple_file.read((char*)tri_arr, sizeof(unsigned int) * 3);
			auto p = make_pair(make_pair(tri_arr[0], tri_arr[2]), tri_arr[1]);
			data_train.push_back(p);
			check_data_train.insert(p);
			check_data_all.insert(p);

			set_entity.insert(tri_arr[0]);
			set_entity.insert(tri_arr[2]);
			set_relation.insert(tri_arr[1]);

			rel_heads[tri_arr[1]][tri_arr[0]]
				.push_back(tri_arr[2]);
			rel_tails[tri_arr[1]][tri_arr[2]]
				.push_back(tri_arr[0]);
			rel_finder[make_pair(tri_arr[0], tri_arr[2])]
				= tri_arr[1];
		}
		triple_file.close();
	}

	void load_testing(	
		const string& filename, 
		vector<pair<pair<int, int>,int>>& vin_true,
		vector<pair<pair<int, int>,int>>& vin_false,
		bool self_sampling = false)
	{
		ifstream triple_file(filename, ios_base::binary);

		size_t triple_size;
		triple_file.read((char*)&triple_size, sizeof(size_t));
		data_train.reserve(triple_size);
		for (unsigned int i = 0; i < triple_size && triple_file; ++i)
		{
			unsigned int tri_arr[3];
			triple_file.read((char*)tri_arr, sizeof(unsigned int) * 3);
			auto p = make_pair(make_pair(tri_arr[0], tri_arr[2]), tri_arr[1]);
			vin_true.push_back(p);
			check_data_all.insert(p);
			set_entity.insert(tri_arr[0]);
			set_entity.insert(tri_arr[2]);
			set_relation.insert(tri_arr[1]);
		}
		triple_file.close();
	}

	void sample_false_triplet(	
		const pair<pair<int,int>,int>& origin,
		pair<pair<int,int>,int>& triplet) const
	{

		double prob = relation_hpt[origin.second]/(relation_hpt[origin.second] + relation_tph[origin.second]);

		triplet = origin;
		while(true)
		{
			if(rand()%1000 < 1000 * prob)
			{
				triplet.first.second = rand()%set_entity.size();
			}
			else
			{
				triplet.first.first = rand()%set_entity.size();
			}

			if (check_data_train.find(triplet) == check_data_train.end())
				return;
		}
	}

	void sample_false_triplet_relation(	
		const pair<pair<int,int>,int>& origin,
		pair<pair<int,int>,int>& triplet) const
	{

		double prob = relation_hpt[origin.second]/(relation_hpt[origin.second] + relation_tph[origin.second]);

		triplet = origin;
		while(true)
		{
			if (rand()%100 < 50)
				triplet.second = rand()%set_relation.size();
			else if (rand() % 1000 < 1000 * prob)
			{
				triplet.first.second = rand() % set_entity.size();
			}
			else
			{
				triplet.first.first = rand() % set_entity.size();
			}

			if (check_data_train.find(triplet) == check_data_train.end())
				return;
		}
	}
};