import numpy as np

class threeparttable:
    """ Print regression tables to latex; requires threeparttable package. """

    def __init__(self, reg_list, 
        filename = 'reg.tex', 
        path = './results/',
        caption = 'Regression Table',
        label = 'tab:reg',
        stretch = 1.05,
        additional_notes = '',
        size = None,
        format_names = True):

        """ Print regression tables to latex; requires threeparttable package. """

        # Create formatters
        real_formatter = "{:,.3f}".format
        int_formatter = "{:,.0f}".format

        # Get the intersection of variables :: XXXXX
        J = len(reg_list)
        keep = 'intersection'
        vars_list = reg_list[0].vars
        if format_names is True:
            vars_list = [ (i.replace('_',' ')).title() for i in vars_list]

        #for j in range(J):
        K = len(vars_list)

        # Track inference structure
        se_types = []
        se_types.append('Standard Errors:')
        se_N_clstr = []
        se_N_clstr.append('N Clusters:')
        se_var_clstr = []
        se_var_clstr.append('Cluster Variable:')
        rbst = 0 
        clstr_rbst = 0
        
        for j in range(J):
            if j>0:
                se_types.append(' & ')
                se_types.append(reg_list[j-1].se_type)
                if reg_list[j-1].se_type =='Robust':
                    rbst =1
                    se_N_clstr.append(' & ')
                    se_var_clstr.append(' & ')
                elif reg_list[j-1].se_type =='Cluster-Robust':
                    clstr_rbst=1
                    se_N_clstr.append(' & ')
                    se_N_clstr.append(int_formatter(reg_list[j-1].n_clusters))
                    se_var_clstr.append(' & ')
                    se_var_clstr.append(reg_list[j-1].cluster_varname)
                else:
                    se_N_clstr.append(' & ')
                    se_var_clstr.append(' & ')
        se_types.append('\\\\ \n')
        se_type_row = ''.join(se_types)
        se_N_clstr.append('\\\\ \n')
        se_N_row = ''.join(se_N_clstr)
        se_var_clstr.append('\\\\ \n')
        se_var_row = ''.join(se_var_clstr)

        ## Make the latex table
        file = open(path+filename, "w")

        # Begin tabular
        
        file.write('\\begin{table} \n')
        file.write('\\centering \n')
        file.write('\\begin{threeparttable}[t] \n')
        file.write('\\caption{'+caption+'} \n')
        file.write('\\label{'+label+'} \n')

        # Begin table
        file.write('\\renewcommand{\\arraystretch}{'+str(stretch)+'}')        
        if not size is None:
            file.write('\\begin{'+size+'+} \n')    

        file.write('\\begin{tabular}{'+(J+1)*'c'+'} ')
        file.write('\\hline \\hline \n')

        # Title row
        title = []
        depvar = []
        for j in range(J+1):
            if j>0:
                title.append(' & ')
                title.append('('+str(j)+')')            
                depvar.append(' & ')
                depvar.append( (reg_list[j-1].depvar).replace('_',' ').title())            
        title.append('\\\\')
        depvar.append('\\\\')
        title_row = ''.join(title)
        file.write(title_row)
        depvar_row = ''.join(depvar)
        file.write(depvar_row)
        file.write('\\hline \n')

        # Point estimates and standard errors
        for k in range(K):
            est = []
            est.append(vars_list[k])
            se = []
            for j in range(J):
                est.append(' & ')
                # Stars for inference
                this_pval = reg_list[j].pval[k]
                if this_pval <= .01:
                    stars = '$^{***}$'
                elif this_pval <= .05:
                    stars = '$^{**}$'
                elif this_pval <= .1:
                    stars = '$^{*}$'
                else:
                    stars = ''
                est.append(str(real_formatter(reg_list[j].beta[k]))+stars)
                ##
                se.append(' & ')
                se.append('(')
                se.append(str(real_formatter(reg_list[j].se[k])))
                se.append(')')
            est.append('\\\\ \n')
            se.append('\\\\ \n')
            # Join lists
            est_row = ''.join(est)
            se_row = ''.join(se)
            file.write(est_row)
            file.write(se_row)
        file.write('\\hline \n')

        ## Additional table rows
        # Sample size
        est = []
        est.append('N Obs.')
        for j in range(J):
            est.append(' & ')
            est.append(str(int_formatter(reg_list[j].n)))
        est.append('\\\\ \n')
        est_row = ''.join(est)
        file.write(est_row)
        # rsquared
        est = []
        est.append('R$^2$')
        for j in range(J):
            est.append(' & ')
            est.append(str(real_formatter(reg_list[j].rsq)))
        est.append('\\\\ \n')
        est_row = ''.join(est)
        file.write(est_row)
        # Mean depvar
        est = []
        est.append('Mean, Dep.Var.')
        for j in range(J):
            est.append(' & ')
            est.append(str(real_formatter(reg_list[j].ybar)))
        est.append('\\\\ \n')
        est_row = ''.join(est)
        file.write(est_row)
        # Fancy SE
        if rbst == 1 or clstr_rbst ==1:
            file.write('\\hline \n')
            file.write(se_type_row)
        if clstr_rbst == 1:
            file.write(se_var_row)
            file.write(se_N_row)

        stars_note = 'Stars denote: $^{*}$:$p<0.10$, $^{**}$:$p<0.05$, $^{***}$:$p<0.01$.'

        file.write('\\hline \n')
        # End tabular section:
        file.write('\\end{tabular} \n ')
        # End notes:
        file.write('\\begin{tablenotes} \n')
        file.write('\small \n')
        file.write('\\item Notes: ')
        file.write(''+additional_notes+stars_note+'\n')
        file.write('\\end{tablenotes} \n ')
        if not size is None:
            file.write('\\end{'+size+'+} \n')    
        file.write('\\end{threeparttable} \n')
        file.write('\\end{table} \n')


        # Close file
        file.close()