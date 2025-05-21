/*****************************************************************
 
    This file is part of the eqtools package.

    EqTools is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    EqTools is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with EqTools.  If not, see <http://www.gnu.org/licenses/>.

    Copyright 2025 Ian C. Faust

******************************************************************/

long ismonotonic(double val[], int ix);
long isregular(double val[], int ix);

void nonreg_ev(double val[], double x0[], double x1[], double x2[], double f[], double fx0[], double fx1[], double fx2[], int ix0, int ix1, int ix2, int ix);
void nonreg_ev_full(double val[], double x0[], double x1[], double x2[], double f[], double fx0[], double fx1[], double fx2[], int ix0, int ix1, int ix2, int ix, int d0, int d1, int d2);
void reg_ev(double val[], double x0[], double x1[], double x2[], double f[], double fx0[], double fx1[], double fx2[], int ix0, int ix1, int ix2, int ix);
void reg_ev_full(double val[], double x0[], double x1[], double x2[], double f[], double fx0[], double fx1[], double fx2[], int ix0, int ix1, int ix2, int ix, int d0, int d1, int d2);
void ev(double val[], double x0[], double x1[], double x2[], double f[], double fx0[], double fx1[], double fx2[], int ix0, int ix1, int ix2, int ix);
