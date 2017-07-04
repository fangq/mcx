%define codename %NAME%

Name:           %{codename}
Version:        %VERSION%
Release:        1%{?dist}
Summary:        MCX Studio - GUI for Monte Carlo eXtreme

Group:          Applications/Engineering

License:        GPLv3
URL:            http://mcx.space
Source0:        http://downloads.sourceforge.net/wqy/%{codename}-%{version}.tar.gz
BuildRoot:      %(mktemp -ud %{_tmppath}/%{name}-%{version}-%{release}-XXXXXX)

BuildArch:      x86_64
BuildRequires:  lazarus
Requires:       mcx

%description
%INFO%

%prep
%setup -q -n %{codename}


%build

make %{codename}


%install
rm -fr %{buildroot}

install -m 0755 -d %{buildroot}%{_libdir}
install -m 0644 -p %{codename} %{buildroot}%{_libdir}
install -m 0755 -d %{buildroot}%{_bindir}


%clean
rm -fr %{buildroot}


%doc AUTHORS.txt ChangeLog.txt LICENSE.txt README.txt
%attr(744 ,root ,root) %{_bindir}/%{setscript}


%changelog
%changelog
* Tue Jul 04 2017 Qianqian Fang <fangqq at gmail.com>
- Initial RPM release

