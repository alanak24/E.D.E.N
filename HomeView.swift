//
//  HomeView.swift
//  EDEN
//
//  Created by Alana Kumar on 30/4/2026.
//

import SwiftUI

struct HomeView: View {
    @StateObject var viewModel = MovieViewModel()
    @StateObject var tvViewModel = TVViewModel()
    @EnvironmentObject var movieVM: MovieViewModel
    @EnvironmentObject var showVM: TVViewModel
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            VStack(spacing: 20) {

                Image("edenlogo")
                    .resizable()
                    .scaledToFit()
                    .frame(width: 220, height: 220)

                Text("Popular movies")
                    .foregroundColor(.white)

                // Horizontal scrolling movies
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 16) {

                        ForEach(viewModel.movies) { movie in
                            NavigationLink(destination: MovieDetailView(movie: movie)
                                .environmentObject(viewModel)){
                                VStack {
                                    
                                    AsyncImage(
                                        url: URL(string: "https://image.tmdb.org/t/p/w500\(movie.poster_path)")
                                    ) { image in
                                        image
                                            .resizable()
                                            .scaledToFill()
                                    } placeholder: {
                                        ProgressView()
                                    }
                                    .frame(width: 120, height: 180)
                                    .cornerRadius(12)
                                    
                                    Text(movie.title)
                                        .font(.caption)
                                        .foregroundColor(.white)
                                        .lineLimit(1)
                                }
                            }
                        
                        }
                    }
                    .padding(.horizontal)
                }
                // MARK: - TV SHOWS
                Text("Popular TV Shows")
                    .foregroundColor(.white)
                    .padding(.horizontal)

                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 16) {
                        ForEach(tvViewModel.shows) { show in
                            NavigationLink(destination: TVDetailView(show: show) .environmentObject(tvViewModel)) {
                                VStack {
                                    AsyncImage(
                                        url: URL(string:
                                            "https://image.tmdb.org/t/p/w500\(show.poster_path)"
                                        )
                                    ) { image in
                                        image.resizable().scaledToFill()
                                    } placeholder: {
                                        ProgressView()
                                    }
                                    .frame(width: 120, height: 180)
                                    .cornerRadius(12)

                                    Text(show.name)
                                        .font(.caption)
                                        .foregroundColor(.white)
                                        .lineLimit(1)
                                }
                            }
                        }
                    }
                    .padding(.horizontal)
                }
                

                Spacer()
            }
            .padding(.top)
        }
        .navigationBarBackButtonHidden(true)
        .onAppear {
            viewModel.fetchMovies()
            tvViewModel.fetchShows()
        }
    }
}


#Preview {
    HomeView()
}
