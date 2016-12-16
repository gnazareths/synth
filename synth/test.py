pred = pd.DataFrame({'a':[3,1,6,1], 'b':[4,2,5,0], 'c':[5,2,3,5],
                         'd':[4,2,4,2], 'e': [3,4,7,2]},index=['A','B','C','D'])

outc = pd.DataFrame({'a':[11,10,12,13,13,12,13], 'b':[8,8,10,11,11,11,12],
                         'c':[20,21,25,27,27,28,29], 'd':[16,17,22,25,25,26,27],
                         'e':[14,14,17,20,21,21,23]}, 
                         index=[2010,2011,2012,2013,2014,2015,2016])
    
output = synth_tables( pred,
                       outc,
                       'a',
                       ['b','c','d','e'],
                       ['A','B','C','D','E'],
                       [2010,2011,2012,2013,2014],[2010,2011,2012,2013,2014,2015,2016]
                     )
output

def plot(synth_tables):
    estimates, actual_values = synth_tables[0], synth_tables[1]
    plt.plot(range(len(estimates)),estimates, 'r--', label="Synthetic Control")
    plt.plot(range(len(estimates)),actual_values, 'b-', label="Actual Data")
    plt.title("Example Synthetic Control Model")
    plt.ylabel("Y axis")
    plt.xlabel("X axis")
    plt.legend(loc='upper left')
    plt.show()

plot(output)
